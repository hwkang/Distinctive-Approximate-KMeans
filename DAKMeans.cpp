/*
***************************************
*** Distinctive Approximate K-Means ***
***************************************
Author: Hongwen Henry Kang (hongwenk@cs.cmu.edu)

Implementation of:
[1] Hongwen Kang, Martial Hebert, Takeo Kanade. "Image Matching with Distinctive Visual Vocabulary", In Proceedings of the IEEE Workshop on Applications of Computer Vision 2011 (WACV 2011). Kona, Hawaii, 2011.

Usage: ./DAKMeans feats.txt num_feat dim_feat K output_dir [cluster_output.txt] [memberof_output.txt] [flann_output.txt] [num_thread] [num_trial] [feat_select.txt] [lambda] [Rp] [n] [cluster_stats.txt] [weight_file] [cluster_init]

Software features:
1. Approximate nearest neighbor in cluster assignment (using FLANN library)
2. Multiple threading (using pthread library)
3. Measuring statistics of K-Means to find distictive clusters [1]
4. Select portions of features points from the input feature file (default: use all features)
5. Apply different weightings to the selected features points (default: equally weighted)
6. Allows for discarding feature points with slack variable lambda

Input format:
1. Feature file: [nxp]
n rows of p space separated float numbers, each row corresponds to a feature point

2. Select file: [nx1]
n rows of binary 0|1 numbers, feature i is selected if row i is 1, otherwise ignored
Total number of 1s: m
default: m=n

3. Weight file: [nx1]
n rows of float values, each corresponding to the weight of a feature

4. number of clusters, K: [1x1]

5. num_thread: [1x1]
number of threads to use. Should be equal to or less than number of clusters.
default: 5

6. num_trial: [1x1]
number of trials to run K-means, the solution is updated after each trial if the energy reduces
default: 5

7. lambda: [1x1] or "null"
Slacking variable for discarding feature points, if the minimum distance of a feature point to the cluster centers is greater than lambda, it is discarded.
default: DBL_MAX, i.e. no feature points will be discarded
"null" means using default value

8. Rp [1x1] and n [1x1]
see paper [1] for details

9. cluster_init: "init"
Flag to check whether we should use initialization used before.  This is to garauntee that the initializations are the same between different runs of the program, since K-means solution depends on initialization.  This is a good practice to compare results with different parameters.

Output files format:
1. cluster file: [Kxp]
K rows of p space separated float numbers, each row corresponds to a cluster center

2. memberof file: [mx1]
m rows of integers ([0, K-1]), each corresponds to the cluster index of a feature point

3. flann file: see FLANN library document

4. cluster stats file: [Kxdim_stats]
dim_stats=23
K rows of statistics for K-means clusters, see [1] for details
In each row, S:
S[0], distance of cluster center to its 1-nearest neighbor: d_{NN}
S[1], number of neighbors within distance: Nc=Rp*d_{NN}
S[2], distinctiveness statistics: P=(1-1/(Rp)^n)^Nc
S[2*i+1], feature id of the ith nearest neighbor, i=1,2,...,10
S[2*i+2], distance of the cluster center to the ith nearest neighbor, i=1,2,...,10

*Can be used as a quick and dirty way to calculate approximate K-means clustering
*Please cite [1] if you are using this software.

--
Hongwen Henry Kang
June 13, 2012
www.hwkang.com
--
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>
#include <float.h>
#include <assert.h>

#include "flann.h"
//#include "constants.h"
#include "util.h"

#include "pthread.h"

using namespace std;

extern void saveFLANNParameters(FLANNParameters& ann_para, char* filename);

extern void loadFLANNParameters(FLANNParameters& ann_para, char* filename);

extern void printFLANNParameters(FLANNParameters& ann_para, FILE* stream);

extern bool file_exists(const char * filename);

extern void writeMatrix(float* matrix, size_t rows, size_t cols, char* filename);
extern void loadMatrix(float* matrix, size_t rows, size_t cols, char* filename);
extern void writeMatrix(int* matrix, size_t rows, size_t cols, char* filename);

extern size_t min(size_t a, size_t b);

void randInit(size_t seed, float* db_feats, float* cluster_feats, int* memberof, size_t num_feats, int dim_feat, int num_cluster, std::vector<size_t>* members);

double DAKMeans(float* db_feats, float* cluster_feats, int* memberof, size_t num_feats, int dim_feat, int* num_cluster, int max_iter, int cur_iter, FLANNParameters* ann_para, float speedup, char* output_dir, double* weight, double lambda, int num_thread, float* cluster_stats, int dim_stats, float Rp, int n, bool cluster_inited);

void DAKMeans_assign_master(float* db_feats, float* cluster_feats, int* memberof, size_t num_feats, int dim_feat, int* num_cluster, FLANNParameters *ann_para, float speedup, std::vector<size_t>* members, double* weight, double lambda, int num_thread, double* distance);
void DAKMeans_cluster_master(float* db_feats, float* cluster_feats, int* memberof, size_t num_feats, int dim_feat, int* num_cluster, std::vector<size_t>* members, double* weight, double lambda, int num_thread);

int main(int argc, char **argv)
{
    if (argc < 6 || argc > 18) {
      printf("Usage: %s feats.txt num_feat dim_feat K output_dir [cluster_output.txt] [memberof_output.txt] [flann_output.txt] [num_thread] [num_trial] [feat_select.txt] [lambda] [Rp] [n] [cluster_stats.txt] [weight_file] [cluster_init]\n", argv[0]);
      return 1;
    }

    int argi=1;
    const char *feat_file= argv[argi++];
    // number of features in the input file
    size_t num_input_feat=atoi(argv[argi++]);
    // number of dimensions per feature point
    int dim_feat=atoi(argv[argi++]);
    // number of clusters in K-means
    int K=atoi(argv[argi++]);

    bool *selected = NULL;
    FILE* select_fp = NULL;
    size_t nselected=0;

    // number of features eventually selected (default: num_input_feat)
    size_t num_db_feat;

    // try multiple times and pick the one with the minimum energy
    int num_trial = 5; // 5 is my magic number

    // using multiple threads
    int num_thread = 5; // 5 is my magic number

    // dimension of distinctive statistics, ignore this if you are just using it as a way to get kmeans
    int dim_stats = 23;

    // parameter for distinctive statistics, see paper [1]
    double lambda = DBL_MAX;
    float Rp=1.8447;
    int n=10;

    char cluster_stats_file[256]="cluster_stats.txt";
    const char *weight_file = NULL;
    bool cluster_inited=false;

    fprintf(stdout, "num_input_feat: %ld\n", num_input_feat);
    fprintf(stdout, "dim_feat=%d\n", dim_feat);
    fflush(stdout);

    char output_dir[256] = "";
    strcpy(output_dir, argv[argi++]);

    char cluster_output_file[256]= "cluster.txt";
    if (argc>=7){
      strcpy(cluster_output_file, argv[argi++]);
    }
    char memberof_output_file[256]="memberof.txt";
    if (argc>=8){
      strcpy(memberof_output_file, argv[argi++]);
    }
    char flann_output_file[256]="flann_output.txt";
    if (argc>=9){
      strcpy(flann_output_file, argv[argi++]);
    }

    // mkdir output_dir if not exist
    struct stat st;
    if(stat(output_dir, &st)!=0){
      int t=umask(0);
      mkdir(output_dir, 0777);
      umask(t);
    }

    char temp[256]="";
    strcpy(temp, output_dir);
    strcat(temp, "/");
    strcat(temp, memberof_output_file);
    strcpy(memberof_output_file, temp);

    strcpy(temp, "");
    strcpy(temp, output_dir);
    strcat(temp, "/");
    strcat(temp, cluster_output_file);
    strcpy(cluster_output_file, temp);

    strcpy(temp, "");
    strcpy(temp, output_dir);
    strcat(temp, "/");
    strcat(temp, flann_output_file);
    strcpy(flann_output_file, temp);

    fprintf(stdout, "output_dir = %s\n", output_dir);
    fprintf(stdout, "memberof_output_file = %s\n", memberof_output_file);
    fprintf(stdout, "cluster_output_file = %s\n", cluster_output_file);
    fprintf(stdout, "flann_output_file = %s\n", flann_output_file);

    fflush(stdout);

    if (argc>=10){
      num_thread = atoi(argv[argi++]);
    }
    num_thread = min(num_thread, K); // We are splitting the jobs based on clusters, we do not split the job more than the number of clusters
    fprintf(stdout, "num_thread: %d\n", num_thread);

    if (argc>=11){
      num_trial = atoi(argv[argi++]);
    }
    fprintf(stdout, "num_trial: %d\n", num_trial);

    char select_file[256]="null";

    if (argc>=12){
      strcpy(select_file, argv[argi++]);
    }

    if (argc>=13){
      char* arg=argv[argi++];
      if(strcmp(arg, "null"))
	lambda = atof(arg);
      fprintf(stdout, "lambda: %f\n", lambda);
    }

    if (argc>=15){
      Rp=atof(argv[argi++]);
      n=atoi(argv[argi++]);
      fprintf(stdout, "Rp: %f\nn: %d\n", Rp, n);
    }

    if (argc>=16){
      strcat(cluster_stats_file, argv[argi++]);
    }
    strcpy(temp, output_dir);
    strcat(temp, "/");
    strcat(temp, cluster_stats_file);
    strcpy(cluster_stats_file, temp);
    fprintf(stdout, "cluster_stats_file: %s\n", cluster_stats_file);

    if (argc>=17){
      weight_file = argv[argi++];
      fprintf(stdout, "weight_file: %s\n", weight_file);
    }

    if (argc>=18){
      if(0==strcmp(argv[argi++], "init"))
	 cluster_inited = true;
      fprintf(stdout, "use previously generated cluster_init file\n");
    }

    double start, end, start1, end1;
    double *weight = NULL;

    float *db_feats=NULL;
    FILE* feat_fp = NULL;
    float *cluster_feats=NULL;
    int* memberof = NULL;
    float *cluster_stats=NULL;
    float *tmp_cluster_feats=NULL;
    int* tmp_memberof = NULL;
    float *tmp_cluster_stats=NULL;

    // maximum number of iterations per each trial
    int max_iter = 10;

    // variables for ANN
    FLANN_INDEX ann_index=NULL;
    FLANNParameters ann_para;
    float speedup = 0;

    int* ann_indices = NULL;
    float* ann_dists = NULL;

    size_t cnt = 0;
    double minEnergy = DBL_MAX;

    start = clock();

    cluster_feats=new float [K*dim_feat];
    if (NULL==cluster_feats){
      fprintf(stdout, "Error allocating memory for cluster_feats\n");
      goto EXIT;
    }

    tmp_cluster_feats=new float [K*dim_feat];
    if (NULL==tmp_cluster_feats){
      fprintf(stdout, "Error allocating memory for tmp_cluster_feats\n");
      goto EXIT;
    }

    cluster_stats=new float [K*dim_stats];
    if (NULL==cluster_stats){
      fprintf(stdout, "Error allocating memory for cluster_stats\n");
      goto EXIT;
    }

    tmp_cluster_stats=new float [K*dim_stats];
    if (NULL==tmp_cluster_stats){
      fprintf(stdout, "Error allocating memory for tmp_cluster_stats\n");
      goto EXIT;
    }

    // read select file
    nselected = num_input_feat;
    selected = new bool[num_input_feat];
    if(NULL==selected){
      fprintf(stdout, "Unable to allocate memory for selected\n");
      goto EXIT;
    }
    for(size_t i=0; i<num_input_feat; i++) selected[i]=true;
    if(strcmp(select_file, "null")){
      nselected = 0;
      select_fp=fopen(select_file, "r");
      if (select_fp == NULL) {
	fprintf(stdout, "Error opening file %s for reading\n", select_file);
	goto EXIT;
      }
      for(size_t i=0; i<num_input_feat; i++) selected[i]=false;
      size_t si = 0;
      while(fscanf(select_fp, "%ld", &si)!=EOF){
	if (si>0 && si<=num_input_feat){
	  selected[si-1] = true;
	  nselected++;
	}
	si = 0;
      }
      if(select_fp!= NULL)
	fclose(select_fp);
      select_fp=NULL;
    }

    num_db_feat = nselected;
    fprintf(stdout, "num_db_feat: %ld\n", num_db_feat);
    fprintf(stdout, "Now allocate memory (%ldBytes=%.3fGB) for db_feats\n", sizeof(float)*num_db_feat*dim_feat, (double(sizeof(float)*num_db_feat*dim_feat)/1024/1024/1024));
    fflush(stdout);
    db_feats=new float [nselected*dim_feat];

    if (NULL==db_feats){
      fprintf(stdout, "Error allocating memory for db_feats\n");
      goto EXIT;
    }

    // read feature file
    start1 = clock();
    fprintf(stdout, "Reading feature file\n");
    feat_fp=fopen(feat_file, "r");
    if (feat_fp == NULL) {
      fprintf(stdout, "Error opening file %s for reading\n", feat_file);
      goto EXIT;
    }
    cnt = 0;
    for(size_t i=0; i<num_input_feat; i++){
      for(int j=0; j<dim_feat; j++){
	float f=0.0;
	fscanf(feat_fp, "%f", &f);
	if(selected[i])
	  *(db_feats+cnt*dim_feat+j)=f;
      }
      if(selected[i])
	cnt++;
    }
    if(feat_fp!= NULL)
      fclose(feat_fp);
    feat_fp=NULL;
    fprintf(stdout, "\r%.3f%% complete.\n", 100.0);
    fprintf(stdout, "Done\n");
    end1=clock();
    printf("Time used in reading feature file: %0.5fs.\n", double(end1 - start1) / ((double) CLOCKS_PER_SEC));
    fflush(stdout);

    memberof = new int[num_db_feat];
    if (NULL==memberof){
      fprintf(stdout, "Error allocating memory for memberof\n");
      goto EXIT;
    }

    tmp_memberof = new int[num_db_feat];
    if (NULL==tmp_memberof){
      fprintf(stdout, "Error allocating memory for tmp_memberof\n");
      goto EXIT;
    }

    // read weight file
    if (NULL!=weight_file && strcmp(weight_file, "null")){
      fprintf(stdout, "Read weight_file: %s\n", weight_file);
      FILE* weight_fp=fopen(weight_file, "r");
      if (weight_fp == NULL) {
	fprintf(stdout, "Error opening file %s for reading\n", weight_file);
	goto EXIT;
      }
      weight = new double[num_db_feat];
      if (NULL==weight){
	fprintf(stdout, "Unable to allocate memory for weight\n");
	fflush(stdout);
	goto EXIT;
      }
      cnt = 0;
      for(size_t i=0; i<num_input_feat; i++){
	float f=0.0;
	fscanf(weight_fp, "%f", &f);
	if(selected[i]){
	  *(weight+cnt)=f;
	  cnt++;
	}
      }
      if(weight_fp!= NULL)
	fclose(weight_fp);
      weight_fp=NULL;
      fprintf(stdout, "Done\n");
    }

    // default ANN parameters
    // TODO is this the right default?
    ann_para.algorithm = FLANN_INDEX_KDTREE;
    ann_para.target_precision = 0.9;
    ann_para.build_weight = 0.01;
    ann_para.memory_weight = 0.1;
    ann_para.sample_fraction = 0.2;
    ann_para.sample_fraction = float(min(max(min(100000, size_t(num_db_feat*0.2)), 10000), num_db_feat))/num_db_feat; //using the minimum of 100000 features or 20% input features for K-means
    //ann_para.log_destination = NULL;

    ann_index = NULL;

    if (file_exists(flann_output_file)){
      // load ANN parameters if exists
      cout<<"Loading FLANN parameters."<<endl;
      fflush(stdout);
      fflush(stdout);

      loadFLANNParameters( ann_para, flann_output_file);

      cout<<"FLANN parameters loaded."<<endl;
      fflush(stdout);
      fflush(stdout);

      //ann_para.log_destination = NULL;
      ann_para.log_level = FLANN_LOG_ERROR;

      printFLANNParameters(ann_para, stdout);
      fflush(stdout);
      fflush(stdout);
    }
    else{
      // build ANN index if not exists
      ann_index = flann_build_index(db_feats, num_db_feat, dim_feat, &speedup, &ann_para);

      if (NULL == ann_index){
	fprintf(stdout, "Fail to build flann index\n");
	fflush(stdout);
	fflush(stdout);
	goto EXIT;
      }
      saveFLANNParameters( ann_para, flann_output_file);
    }

    ann_para.target_precision = -1;// use for built index
    //ann_para.log_level = FLANN_LOG_ERROR;
    //ann_para.log_level = FLANN_LOG_INFO;
    ann_para.log_level = FLANN_LOG_WARN;

    for(int itrial=0; itrial<num_trial; itrial++){

      max_iter = 10;

      fprintf(stdout, "start DAKMeans with new random start\n");

      // Run DAKmeans once
      double energy = DAKMeans(db_feats, tmp_cluster_feats, tmp_memberof, num_db_feat,  dim_feat, &K, max_iter, itrial, &ann_para, speedup, output_dir, weight, lambda, num_thread, tmp_cluster_stats, dim_stats, Rp, n, cluster_inited);

      char filename[256];
      char output_prefix[256]="";
      if(NULL!=output_dir)
	strcpy(output_prefix, output_dir);
      sprintf(filename, "%s/memberof%03d.txt", output_prefix, itrial);
      fprintf(stdout, "write tmp_memberof to %s\n", filename);
      fflush(stdout);
      writeMatrix(tmp_memberof, num_db_feat, 1, filename);

      sprintf(filename, "%s/cluster%03d.txt", output_prefix, itrial);
      fprintf(stdout, "write tmp_cluster_feats to %s\n", filename);
      fflush(stdout);
      writeMatrix(tmp_cluster_feats, K, dim_feat, filename);

      sprintf(filename, "%s/cluster_stats%03d.txt", output_prefix, itrial);
      fprintf(stdout, "write cluster_stats to %s\n", filename);
      fflush(stdout);
      writeMatrix(tmp_cluster_stats, K, dim_stats, filename);

      // update clustering if energy reduces
      if(energy<minEnergy){
	minEnergy = energy;
	memcpy(cluster_feats, tmp_cluster_feats, sizeof(float)*K*dim_feat);
	memcpy(memberof, tmp_memberof, sizeof(int)*num_db_feat);
	memcpy(cluster_stats, tmp_cluster_stats, sizeof(float)*K*dim_stats);

	fprintf(stdout, "write memberof to %s\n", memberof_output_file);
	fflush(stdout);
	writeMatrix(memberof, num_db_feat, 1, memberof_output_file);

	fprintf(stdout, "write cluster_feats to %s\n", cluster_output_file);
	fflush(stdout);
	writeMatrix(cluster_feats, K, dim_feat, cluster_output_file);

	if(strcmp("", cluster_stats_file)){
	  fprintf(stdout, "write cluster_stats to %s\n", cluster_stats_file);
	  fflush(stdout);
	  writeMatrix(cluster_stats, K, dim_stats, cluster_stats_file);
	}
      }
      fprintf(stdout, "The minEnergy at round %d is: %f\n", itrial, minEnergy);
      fflush(stdout);
    }

    fprintf(stdout, "write memberof to %s\n", memberof_output_file);
    fflush(stdout);
    writeMatrix(memberof, num_db_feat, 1, memberof_output_file);

    fprintf(stdout, "write cluster_feats to %s\n", cluster_output_file);
    fflush(stdout);
    writeMatrix(cluster_feats, K, dim_feat, cluster_output_file);

    if(strcmp("", cluster_stats_file)){
      fprintf(stdout, "write cluster_stats to %s\n", cluster_stats_file);
      fflush(stdout);
      writeMatrix(cluster_stats, K, dim_stats, cluster_stats_file);
    }

    end = clock();
    printf("Total time consumed: %0.5fs.\n", double(end - start) / ((double) CLOCKS_PER_SEC));

 EXIT:
    if(feat_fp!= NULL)
      fclose(feat_fp);
    feat_fp=NULL;

    if(NULL!=db_feats){
      delete[] db_feats;
      db_feats=NULL;
    }

    if(NULL!=cluster_feats){
      delete[] cluster_feats;
      cluster_feats=NULL;
    }

    if(NULL!=tmp_cluster_feats){
      delete[] tmp_cluster_feats;
      tmp_cluster_feats=NULL;
    }

    if(NULL!=cluster_stats){
      delete[] cluster_stats;
      cluster_stats=NULL;
    }

    if(NULL!=tmp_cluster_stats){
      delete[] tmp_cluster_stats;
      tmp_cluster_stats=NULL;
    }

    if(NULL!=memberof){
      delete[] memberof;
      memberof=NULL;
    }

    if(NULL!=tmp_memberof){
      delete[] tmp_memberof;
      tmp_memberof=NULL;
    }

    if (NULL!=ann_indices){
      delete[] ann_indices;
      ann_indices=NULL;
    }

    if (NULL!=ann_dists){
      delete[] ann_dists;
      ann_dists=NULL;
    }

    if (NULL !=ann_index){
      flann_free_index(ann_index, &ann_para);
    }

    if (NULL != weight){
      delete[] weight;
      weight = NULL;
    }

    if (NULL != selected){
      delete[] selected;
      selected = NULL;
    }

    printf("finished\n");

    return 0;

}

void randInit(size_t seed, float* db_feats, float* cluster_feats, int* memberof, size_t num_feats, int dim_feat, int num_cluster, std::vector<size_t>* members){
  /* initialize random seed: */
  srand ( seed );
  bool* init_center=new bool [num_feats];
  int* member_cnt=new int[num_cluster];
  int min_member = 2;
  if (NULL==init_center){
    fprintf(stdout, "Error allocating memory for init_center\n");
    goto EXIT;
  }
  memset(init_center, 0, sizeof(bool)*num_feats);

  if (NULL==member_cnt){
    fprintf(stdout, "Error allocating memory for member_cnt\n");
    goto EXIT;
  }
  for(int i=0; i<num_cluster; i++)
    member_cnt[i]=0;

  // mark all the features that already belong to some clusters
  for(size_t i=0; i<num_feats; i++){
    if(memberof[i]<num_cluster){
      init_center[i]=true;
      member_cnt[memberof[i]]=member_cnt[memberof[i]]+1;
    }
  }

  for(int i=0; i<num_cluster; i++){
    if(member_cnt[i]<min_member){
      printf("cluster[%d] has less than %d members, randomly initiate it\n", i,min_member);
      bool found = false;
      while(!found){
	size_t ri=size_t(rand()%num_feats);
	printf("ri: %d, num_feats: %d\n", ri, num_feats);
	if(ri>=0 && ri<num_feats){
	  if(!init_center[ri]){
	    init_center[ri]=true;
	    for(int j=0; j<dim_feat; j++){
	      *(cluster_feats+i*dim_feat+j)=*(db_feats+ri*dim_feat+j);
	    }
	    members[i].push_back(ri);
	    memberof[ri]=i;
	    found = true;
	  }
	}
      }
    }
  }

  for(size_t i=0; i<num_feats; i++){
    if(!init_center[i]){
      int ri = num_cluster;
      while(ri<0||ri>=num_cluster){
	ri = int(double(rand())/(RAND_MAX)*(num_cluster-1));
      }
      memberof[i]=ri;
      members[ri].push_back(i);
    }
  }

  printf("Finished random initialization\n");

 EXIT:
  if(NULL != init_center)
    delete[] init_center;
  init_center=NULL;
  if(NULL != member_cnt)
    delete[] member_cnt;
}

typedef struct KEYVAL{
  size_t key;
  double val;
};

void priorityQueue(std::vector<KEYVAL >* queue, size_t L, double val, size_t key){
  struct KEYVAL p;
  p.key=key;
  p.val=val;
  size_t i=0;
  for(i=0; i<queue->size(); i++){
    if((*queue)[i].val>val){
      vector<KEYVAL >::iterator it=queue->begin()+i;
      queue->insert(it, p);
      break;
    }
  }
  if(i==queue->size()&&i<L){
    queue->push_back(p);
  }
  while(queue->size()>L){
    queue->pop_back();
  }
}

// generats cluster statistics
void generateStats(float* cluster_stats, int dim_stats, int num_cluster, double* distance, size_t num_feats, int* memberof, std::vector<size_t>* members, float Rp, int n, int L){
  memset(cluster_stats, 0, sizeof(float)*num_cluster*dim_stats);
  for(int i=0; i<num_cluster; i++){
    cluster_stats[i*dim_stats]=DBL_MAX;
  }
  for(size_t i=0; i<num_feats; i++){
    if(memberof[i]<num_cluster){
      if(distance[i]<cluster_stats[memberof[i]*dim_stats] && distance[i]>0.0)
	cluster_stats[memberof[i]*dim_stats] = distance[i];
    }
  }
  for(size_t i=0; i<num_feats; i++){
    if(memberof[i]<num_cluster && distance[i]<cluster_stats[memberof[i]*dim_stats]*Rp && distance[i]>0.0)
	cluster_stats[memberof[i]*dim_stats+1] += 1;
  }
  float Rpn=1-1/pow(Rp, n);
  for(int i=0; i<num_cluster; i++){
    if(cluster_stats[i*dim_stats+1]>0)
      cluster_stats[i*dim_stats+2]=pow(Rpn, cluster_stats[i*dim_stats+1]);

    std::vector<KEYVAL> queue;
    for(size_t j=0; j<members[i].size(); j++){
      priorityQueue(&queue, L, distance[members[i][j]], members[i][j]);
    }

    for(size_t j=0; j<queue.size(); j++){
      assert(j<size_t(L));
      cluster_stats[i*dim_stats+j*2+3]=queue[j].key;
      cluster_stats[i*dim_stats+j*2+4]=queue[j].val;
    }
    queue.clear();
  }
}

double DAKMeans(float* db_feats, float* cluster_feats, int* memberof, size_t num_feats, int dim_feat, int* num_cluster, int max_iter, int cur_try, FLANNParameters *ann_para, float speedup, char* output_dir, double* weight, double lambda, int num_thread, float* cluster_stats, int dim_stats, float Rp, int n, bool cluster_inited)
{
  char output_prefix[256]="";
  if(NULL!=output_dir)
    strcpy(output_prefix, output_dir);

  std::vector<size_t>* members=NULL;

  double energy = 0.0;
  double minEnergy = DBL_MAX;
  int iter=0;

  members = new std::vector<size_t>[(*num_cluster)];

  int* tmp_memberof = NULL;
  float *tmp_cluster_feats = NULL;
  double *distance = NULL;
  float *tmp_cluster_stats = NULL;
  tmp_memberof = new int[num_feats];
  if (NULL==tmp_memberof){
    fprintf(stdout, "Error allocating memory for tmp_memberof\n");
    energy = DBL_MAX;
    goto EXIT;
  }
  tmp_cluster_feats=new float [(*num_cluster)*dim_feat];
  if (NULL==tmp_cluster_feats){
    fprintf(stdout, "Error allocating memory for tmp_cluster_feats\n");
    goto EXIT;
  }
  tmp_cluster_stats=new float [(*num_cluster)*dim_stats];
  if (NULL==tmp_cluster_stats){
    fprintf(stdout, "Error allocating memory for tmp_cluster_stats\n");
    goto EXIT;
  }
  distance = new double[num_feats];
  if (NULL==distance){
    fprintf(stdout, "Error allocating memory for tmp_memberof\n");
    energy = DBL_MAX;
    goto EXIT;
  }

  char filename[256];

  // randomly initialize both cluster centers and membership
  for(int i=0; i<num_feats; i++) tmp_memberof[i]=*num_cluster;
  randInit( time(NULL) + cur_try, db_feats, tmp_cluster_feats, tmp_memberof, num_feats, dim_feat, *num_cluster,  members);

  // if we want to use previously generated cluster centers, for the reason to be consistant for comparison, then we can load these centers
  if(!cluster_inited){
    sprintf(filename, "%s/cluster_init_%03d.txt", output_prefix, cur_try);
    fprintf(stdout, "write cluster_feats to %s\n", filename);
    fflush(stdout);
    writeMatrix(tmp_cluster_feats, *num_cluster, dim_feat, filename);
  }
  else{
    sprintf(filename, "%s/cluster_init_%03d.txt", output_prefix, cur_try);
    loadMatrix(tmp_cluster_feats, *num_cluster, dim_feat, filename);
  }

  fprintf(stdout, "Initialize assignment from random centers\n");
  DAKMeans_assign_master( db_feats, tmp_cluster_feats, tmp_memberof, num_feats, dim_feat, num_cluster, ann_para, speedup, members, weight, DBL_MAX, num_thread, distance);
  sprintf(filename, "%s/memberof_init_%03d.txt", output_prefix, cur_try);
  fprintf(stdout, "write memberof to %s\n", filename);
  fflush(stdout);
  writeMatrix(tmp_memberof, num_feats, 1, filename);

  while(iter<max_iter){

    fprintf(stdout, "DAKMeans %d rounds\n", iter);
    double start, end;
    start = clock();

    // adaptively tightens bounds for removing outliers
    double lambdai=lambda;
    if(lambdai<DBL_MAX/10)
      if(iter<5)
	lambdai=2*(5-iter)*lambda;

    fprintf(stdout, "lambdai: %f\n", lambdai);

    // calculate cluster centers
    DAKMeans_cluster_master( db_feats, tmp_cluster_feats, tmp_memberof, num_feats, dim_feat, num_cluster, members, weight, lambdai, num_thread);

    for(int i=0; i<*num_cluster; i++)
	members[i].clear();

    // update assignment of feature points to cluster centers
    DAKMeans_assign_master( db_feats, tmp_cluster_feats, tmp_memberof, num_feats, dim_feat, num_cluster, ann_para, speedup, members, weight, lambdai, num_thread, distance);
    energy = 0.0;
    double lambda2=lambdai;
    if(lambdai<sqrt(DBL_MAX))
      lambda2 = pow(lambdai, 2);

    for(size_t i=0; i<num_feats; i++){
      double ssei = 0.0;
      if(tmp_memberof[i]<(*num_cluster)){
	ssei = pow(distance[i], 2);
	if(NULL!=weight)
	  ssei *= weight[i];
	energy += ssei;
      }
      else{
	if(NULL!=weight)
	  energy += lambda2*pow(weight[i],2);
	else
	  energy += lambda2;
      }
    }

    end = clock();
    fprintf(stdout, "The energy at round %d is: %f\n", iter, energy);
    fflush(stdout);

    printf("DAKMeans iteration: (%d, %d), clustering time: %0.5fs.\n", cur_try, iter, double(end - start) / ((double) CLOCKS_PER_SEC));

    start = clock();

    int L=(dim_stats-3)/2;

    fprintf(stdout, "Start generate statistics\n");
    fflush(stdout);

    generateStats(tmp_cluster_stats, dim_stats, *num_cluster, distance, num_feats, tmp_memberof, members, Rp, n, L);

    end = clock();
    printf("DAKMeans iteration: (%d, %d), stats time: %0.5fs.\n", cur_try, iter, double(end - start) / ((double) CLOCKS_PER_SEC));

    sprintf(filename, "%s/memberof%03d_%03d.txt", output_prefix, cur_try, iter);
    fprintf(stdout, "write memberof to %s\n", filename);
    fflush(stdout);
    writeMatrix(tmp_memberof, num_feats, 1, filename);

    sprintf(filename, "%s/cluster%03d_%03d.txt", output_prefix, cur_try, iter);
    fprintf(stdout, "write cluster_feats to %s\n", filename);
    fflush(stdout);
    writeMatrix(tmp_cluster_feats, *num_cluster, dim_feat, filename);

    sprintf(filename, "%s/cluster_stats%03d_%03d.txt", output_prefix, cur_try, iter);
    fprintf(stdout, "write cluster_stats to %s\n", filename);
    fflush(stdout);
    writeMatrix(cluster_stats, *num_cluster, dim_stats, filename);

    if(energy<minEnergy){
      minEnergy = energy;
      memcpy(memberof, tmp_memberof, sizeof(int)*num_feats);
      memcpy(cluster_feats, tmp_cluster_feats, sizeof(float)*(*num_cluster)*dim_feat);
      memcpy(cluster_stats, tmp_cluster_stats, sizeof(float)*(*num_cluster)*dim_stats);
    }
    iter++;
  }
  fprintf(stdout, "Minimum energy of this try: %f\n", minEnergy);
  energy = minEnergy;

 EXIT:
  if(NULL!=members){
    for(int i=0; i<*num_cluster; i++)
      members[i].clear();
    delete[] members;
  }
  members=NULL;
  if(NULL!=tmp_memberof){
    delete[] tmp_memberof;
    tmp_memberof=NULL;
  }
  if(NULL!=tmp_cluster_feats){
    delete[] tmp_cluster_feats;
    tmp_cluster_feats=NULL;
  }
  if(NULL!=tmp_cluster_stats){
    delete[] tmp_cluster_stats;
    tmp_cluster_stats=NULL;
  }
  if(NULL!=distance){
    delete[] distance;
    distance=NULL;
  }
  return energy;
}

struct THREADARG{
  FLANN_INDEX ann_index;
  FLANNParameters* ann_para;
  int* ann_indices;
  float* ann_dists;
  int ANN;
  float* db_feats;
  int dim_feat;
  size_t offset;
  size_t num_ele;
  size_t num_top;
  double lambda;
  double* weight;
  int* memberof;
  float* cluster_feats;
  double* distance;
  int num_cluster;
  std::vector<size_t>* members;
  float speedup;
  int threadid;
};

void* DAKMeans_assign_worker(void* arg){

  struct THREADARG* targ = (struct THREADARG*)arg;

  fprintf(stdout, "DAKMeans_assign_worker thread #%d starts\n", targ->threadid);
  /*
    fprintf(stdout, "targ->db_feats: %ld\n", size_t(targ->db_feats));
    fprintf(stdout, "targ->num_ele: %ld\n", size_t(targ->num_ele));
    fprintf(stdout, "targ->ann_index: %ld\n", targ->ann_index);
    fprintf(stdout, "targ->ANN: %ld\n", targ->ANN);
    fprintf(stdout, "targ->ann_para->checks: %ld\n", targ->ann_para->checks);
    fprintf(stdout, "targ->ann_para: %ld\n", targ->ann_para);
  */
  fflush(stdout);

  int* ann_indices = NULL;
  float* ann_dists = NULL;
  ann_indices = new int[targ->num_ele];
  ann_dists = new float[targ->num_ele];
  double start, end;
  /*
    fprintf(stdout, "ann_indices: %ld\n", ann_indices);
    fprintf(stdout, "ann_dists: %ld\n", ann_dists);
  */

  fprintf(stdout, "Now find nearest neighbors\n");
  fflush(stdout);
  start = clock();
  // targ->ANN=1
  flann_find_nearest_neighbors(targ->cluster_feats, targ->num_cluster, targ->dim_feat, targ->db_feats, targ->num_ele, ann_indices, ann_dists, targ->ANN, targ->ann_para);
  // originally ann_dists is the square of the actual Euclidean distances
  for(size_t i=0; i<targ->num_ele; i++){
    ann_dists[i]=sqrt(ann_dists[i]);
  }
  end = clock();
  printf("DAKMeans_assign_worker thread #%d, NN search time : %0.5fs.\n", targ->threadid, double(end - start) / ((double) CLOCKS_PER_SEC));

  start = clock();
  for(size_t i=0; i<targ->num_ele; i++){
    size_t actual_i = i+targ->offset;
    assert(actual_i>=0&&actual_i<targ->num_top);

    double ld=targ->lambda;
    if(NULL!=targ->weight)
      ld*=sqrt(targ->weight[actual_i]);

    targ->distance[actual_i]=ann_dists[i];
    if(i<10){
      printf("thread: %ld ann_dists[%ld]=%f\n", targ->threadid, i, ann_dists[i]);
      printf("cluster[%ld]:", ann_indices[i]);
      for(int j=0; j<targ->dim_feat; j++){
	printf(" %f", targ->cluster_feats[ann_indices[i]*targ->dim_feat+j]);
      }
      printf("\n");
      printf("feat[%ld]:", actual_i);
      for(int j=0; j<targ->dim_feat; j++){
	printf(" %f", targ->db_feats[i*targ->dim_feat+j]);
      }
      printf("\n");
    }

    // if nearest neighbor distance is smaller than ld, assign the actual cluster center
    if(ann_dists[i]<=ld){
      targ->memberof[actual_i]=ann_indices[i];
    }
    else // otherwise, assign it to the outlier
      targ->memberof[actual_i]=targ->num_cluster;
  }
  end = clock();
  printf("DAKMeans_assign_worker thread #%d, member assigning time : %0.5fs.\n", targ->threadid, double(end - start) / ((double) CLOCKS_PER_SEC));

  fprintf(stdout, "Finished nearest neighbor search in DAKMeans_assign_worker thread #%d\n",targ->threadid);

  // EXIT:
  if(NULL!=ann_indices)
    delete[] ann_indices;
  ann_indices = NULL;

  if(NULL!=ann_dists)
    delete[] ann_dists;
  ann_dists = NULL;

  pthread_exit((void*) arg);
};

void DAKMeans_assign_master(float* db_feats, float* cluster_feats, int* memberof, size_t num_feats, int dim_feat, int* num_cluster, FLANNParameters *ann_para, float speedup, std::vector<size_t>* members, double* weight, double lambda, int num_thread, double* distance)
{
  fprintf(stdout, "This is DAKMeans_assign_master\n");
  fflush(stdout);

  //fprintf(stdout, "Lambda: %f\n", lambda);
  //fflush(stdout);
  double start, end;
  start = clock();

  int ANN = 1;

  // threading
  if(num_thread<1)
    num_thread = 1;

  pthread_t* threads=NULL;
  pthread_attr_t attr;
  struct THREADARG* threadargs=NULL;

  threadargs = new struct THREADARG[num_thread];
  threads = new pthread_t[num_thread];
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

  for(int i=0; i<*num_cluster; i++)
      members[i].clear();

  for(int t=0; t<num_thread; t++){

    size_t starti = t*size_t(double(num_feats)/num_thread);
    size_t endi = (t+1)*size_t(double(num_feats)/num_thread);

    if(endi>num_feats)
      endi = num_feats;
    if(starti>num_feats)
      starti = num_feats;
    if(t==num_thread-1&&endi<num_feats)
      endi=num_feats;

    threadargs[t].ann_index = NULL;//ann_index;
    threadargs[t].ann_para = ann_para;
    threadargs[t].ANN = ANN;
    threadargs[t].db_feats = db_feats+starti*dim_feat;
    fflush(stdout);
    threadargs[t].dim_feat = dim_feat;
    threadargs[t].offset = starti;
    threadargs[t].num_ele = endi-starti;
    threadargs[t].num_top = num_feats;
    threadargs[t].lambda = lambda;
    threadargs[t].weight = weight;
    threadargs[t].memberof = memberof;
    threadargs[t].num_cluster = (*num_cluster);
    threadargs[t].cluster_feats = cluster_feats;
    threadargs[t].distance = distance;
    threadargs[t].speedup = speedup;
    threadargs[t].threadid = t;

    int rc = pthread_create(&threads[t], &attr, DAKMeans_assign_worker,(void*)&threadargs[t]);

    if (rc) {
      fprintf(stdout, "ERROR; return code from pthread_create() is %d\n", rc);
      exit(-1);
    }

    fprintf(stdout, "Created new thread: #%d\n", t);
    sleep(1);
  }

  pthread_attr_destroy(&attr);
  fprintf(stdout, "Waiting for the working threads to finish\n");
  fflush(stdout);
  for(int t=0; t<num_thread; t++){
    void* status;
    int rc = pthread_join(threads[t], &status);
    if(rc){
      fprintf(stdout, "ERROR; return code from pthread_join() for thread %d is %d\n", t, rc);
      exit(-1);
    }
    fprintf(stdout, "DAKMeans_assign_master: completed join with thread %d with status: %ld\n", t, (long)status);
  }

  fprintf(stdout, "DAKMeans_assign_master: all working threads finished, now let's gather the statistics\n");

  for(size_t i=0; i<num_feats; i++){
    int clusteri = memberof[i];
    if(clusteri<(*num_cluster)&&clusteri>=0)
      members[clusteri].push_back(i);
  }

  end = clock();
  printf("DAKMeans_assign_master, total time: %0.5fs.\n", double(end - start) / ((double) CLOCKS_PER_SEC));

  if(NULL!=threads){
    delete[] threads;
    threads=NULL;
  }

  if(NULL!=threadargs){
    delete[] threadargs;
    threadargs=NULL;
  }
  return;
}

void* DAKMeans_cluster_worker(void* arg){

  double start, end;
  start = clock();
  struct THREADARG* targ = (struct THREADARG*)arg;

  fprintf(stdout, "DAKMeans_cluster_worker thread #%d starts\n", targ->threadid);
  fprintf(stdout, "DAKMeans_cluster_worker thread #%d, targ->num_ele: %ld\n", targ->threadid, size_t(targ->num_ele));
  fflush(stdout);

  float *cur_cluster = NULL;
  cur_cluster = new float[targ->dim_feat];

  if(NULL==cur_cluster){
    fprintf(stdout, "Fail to allocate memory for cur_cluster\n");
    goto EXIT;
  }

  for(size_t i=0; i<targ->num_ele; i++){
    size_t actual_i = i+targ->offset;
    assert(actual_i>=0&&actual_i<targ->num_top);

    memset(cur_cluster, 0, sizeof(float)*targ->dim_feat);
    double total_weight = 0.0;
    double num_members = double(targ->members[actual_i].size());
    if(num_members==0)
      num_members=1.0;
    for(uint j=0; j<targ->members[actual_i].size(); j++){
      size_t feati=targ->members[actual_i][j];
      for(int k=0; k<targ->dim_feat; k++){
	if(NULL!=targ->weight){
	  cur_cluster[k]=cur_cluster[k]+targ->db_feats[feati*targ->dim_feat+k]*targ->weight[feati];
	}
	else{
	  cur_cluster[k]=cur_cluster[k]+(targ->db_feats[feati*targ->dim_feat+k]/num_members);
	}
      }
      if(NULL!=targ->weight)
	total_weight +=targ->weight[feati];
    }
    if(total_weight==0.0) total_weight = 1.0;
    for(int k=0; k<targ->dim_feat; k++){
      cur_cluster[k]=cur_cluster[k]/total_weight;
    }
    if(targ->members[actual_i].size()>0)
      memcpy(targ->cluster_feats+actual_i*targ->dim_feat, cur_cluster, sizeof(float)*targ->dim_feat);
  }

  end = clock();
  printf("DAKMeans_cluster_worker: #%d, total time: %0.5fs.\n", targ->threadid, double(end - start) / ((double) CLOCKS_PER_SEC));

  EXIT:
  if(NULL!=cur_cluster)
    delete[] cur_cluster;
  cur_cluster = NULL;

  pthread_exit((void*) arg);
};

void DAKMeans_cluster_master(float* db_feats, float* cluster_feats, int* memberof, size_t num_feats, int dim_feat, int* num_cluster, std::vector<size_t>* members, double* weight, double lambda, int num_thread){

  fprintf(stdout, "This is DAKMeans_cluster_master\n");
  fflush(stdout);

  double start, end;
  start = clock();

  if(num_thread<1)
    num_thread = 1;

  pthread_t* threads=NULL;
  pthread_attr_t attr;
  struct THREADARG* threadargs=NULL;
  threadargs = new struct THREADARG[num_thread];
  threads = new pthread_t[num_thread];

  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

  for(int t=0; t<num_thread; t++){

    int starti = t*size_t(double(*num_cluster)/num_thread);
    int endi = (t+1)*size_t(double(*num_cluster)/num_thread);

    if(endi>*num_cluster)
      endi = *num_cluster;
    if(starti>*num_cluster)
      starti = *num_cluster;
    if(t==num_thread-1&&endi<*num_cluster)
      endi=*num_cluster;

    threadargs[t].ann_index = NULL;
    threadargs[t].ann_para = NULL;
    threadargs[t].ANN = 1;
    threadargs[t].db_feats = db_feats;
    threadargs[t].dim_feat = dim_feat;
    threadargs[t].offset = starti;
    threadargs[t].num_ele = endi-starti;
    threadargs[t].num_top = *num_cluster;
    threadargs[t].lambda = lambda;
    threadargs[t].weight = weight;
    threadargs[t].memberof = memberof;
    threadargs[t].cluster_feats = cluster_feats;
    threadargs[t].num_cluster = (*num_cluster);
    threadargs[t].members = members;
    threadargs[t].threadid = t;

    int rc = pthread_create(&threads[t], &attr, DAKMeans_cluster_worker,(void*)&threadargs[t]);

    if (rc) {
      fprintf(stdout, "ERROR; return code from pthread_create() is %d\n", rc);
      exit(-1);
    }
  }

  pthread_attr_destroy(&attr);
  fprintf(stdout, "Waiting for the working threads to finish\n");
  fflush(stdout);

  for(int t=0; t<num_thread; t++){
    void* status;
    int rc = pthread_join(threads[t], &status);
    if(rc){
      fprintf(stdout, "ERROR; return code from pthread_join() for thread %d is %d\n", t, rc);
      exit(-1);
    }
    fprintf(stdout, "DAKMeans_cluster_master: completed join with thread %d with status: %ld\n", t, (long)status);
  }

  fprintf(stdout, "DAKMeans_cluster_master: all working threads finished\n");

  end = clock();
  printf("DAKMeans_cluster_master, total time: %0.5fs.\n", double(end - start) / ((double) CLOCKS_PER_SEC));
  // EXIT:
  if(NULL!=threads){
    delete[] threads;
    threads=NULL;
  }

  if(NULL!=threadargs){
    delete[] threadargs;
    threadargs=NULL;
  }
}
