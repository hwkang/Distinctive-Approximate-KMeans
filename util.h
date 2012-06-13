#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <string>
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>
#include <float.h>
#include <map>
#include <assert.h>

#include "flann.h"
#include "constants.h"

using namespace std;

template <typename T>
bool loadMatrix(char* filename, size_t row, size_t col, T* data){
  bool ret = false;
  FILE* fp=NULL;
  fp = fopen(filename, "r");
  if(NULL==fp){
    cerr<<"Failed to open "<<filename<<" for reading"<<endl;
    return false;
  }
  float f=0;
  for(size_t i=0; i<row; i++){
    char c='\0';
    for(size_t j=0; j<col; j++){
      fscanf(fp, "%f%c", &f, &c);
      data[i*col+j]=(T)f;
      if('\n'==c) break;
    }      
    if('\n'!=c){
      cerr<<"error reading "<<filename<<endl;
      fclose(fp);
      fp=NULL;
      return false;	  
    }  
  }
  fclose(fp);
  fp=NULL;
  ret = true;
  return ret;
}

void writeMatrix(float* matrix, size_t rows, size_t cols, char* filename)
{
  if(NULL == matrix){
    fprintf(stdout, "The input matrix is empty\n");
    fflush(stdout);
    return;
  }
  
  FILE* fp=fopen(filename, "wt");
  
  for(size_t i=0; i<rows; i++){
    float f=*(matrix+i*cols);
    fprintf(fp, "%f", f);
    for(size_t j=1; j<cols; j++){
      f=*(matrix+i*cols+j);
      fprintf(fp, " %f", f);
    }
    fprintf(fp, "\n");
  }
  
  if(NULL!=fp)
    fclose(fp);
  fp=NULL;
};

void loadMatrix(float* matrix, size_t rows, size_t cols, char* filename)
{
  if(NULL == matrix){
    fprintf(stdout, "The input matrix is empty\n");
    fflush(stdout);
    return;
  }
  
  FILE* fp=fopen(filename, "r");
  
  for(size_t i=0; i<rows; i++){    
    for(size_t j=0; j<cols; j++){
      float f=0.0;
      fscanf(fp, "%f", &f);
      *(matrix+i*cols+j)=f;
    }
  }
  
  if(NULL!=fp)
    fclose(fp);
  fp=NULL;
};

void writeMatrix(int* matrix, size_t rows, size_t cols, char* filename)
{
  if(NULL == matrix){
    fprintf(stdout, "The input matrix is empty\n");
    fflush(stdout);
    return;
  }
  
  FILE* fp=fopen(filename, "wt");
  
  for(size_t i=0; i<rows; i++){
    int f=*(matrix+i*cols);
    fprintf(fp, "%d", f);
    for(size_t j=1; j<cols; j++){
      f=*(matrix+i*cols+j);
      fprintf(fp, " %d", f);
    }
    fprintf(fp, "\n");
  }
  
  if(NULL!=fp)
    fclose(fp);
  fp=NULL;
};

bool file_exists(const char * filename)
{ 
  if (FILE * file = fopen(filename, "r")) //I'm sure, you meant for READING =)                                                                               
    { 
      fclose(file);
      return true;
    }
  return false;
};

float abs(float x){
  return x*(x>0?1:-1);
}

void saveFLANNParameters(FLANNParameters& ann_para, char* filename){
  FILE* fp = NULL;
  fp = fopen(filename, "wt");
  if(NULL == fp){
    fprintf(stdout, "Fail to open file %s for writing!", filename);
    return;
  }

  fprintf(fp, "%d\n", int(ann_para.algorithm));
  fprintf(fp, "%d\n", ann_para.checks);
  fprintf(fp, "%f\n", ann_para.cb_index);
  fprintf(fp, "%d\n", ann_para.trees);
  fprintf(fp, "%d\n", ann_para.branching);
  fprintf(fp, "%d\n", ann_para.iterations);
  fprintf(fp, "%d\n", int(ann_para.centers_init));
  fprintf(fp, "%f\n", ann_para.target_precision);
  fprintf(fp, "%f\n", ann_para.build_weight);
  fprintf(fp, "%f\n", ann_para.memory_weight);
  fprintf(fp, "%f\n", ann_para.sample_fraction);
  fprintf(fp, "%d\n", int(ann_para.log_level));
  fprintf(fp, "%s\n", ann_para.log_destination);
  fprintf(fp, "%ld\n", ann_para.random_seed);
  fclose(fp);
  fp=NULL;
}

void printFLANNParameters(FLANNParameters& ann_para, FILE* stream){                                                                                          
  fprintf(stream, "%d\n", int(ann_para.algorithm));
  fprintf(stream, "%d\n", ann_para.checks);                                                                                                            
  fprintf(stream, "%f\n", ann_para.cb_index);                                                                                                           
  fprintf(stream, "%d\n", ann_para.trees);                                                                                                               
  fprintf(stream, "%d\n", ann_para.branching);                                                                                                            
  fprintf(stream, "%d\n", ann_para.iterations);                                                                                                           
  fprintf(stream, "%d\n", int(ann_para.centers_init));                                                                                                     
  fprintf(stream, "%f\n", ann_para.target_precision);                                                                                                     
  fprintf(stream, "%f\n", ann_para.build_weight);                                                                                                         
  fprintf(stream, "%f\n", ann_para.memory_weight);                                                                                                        
  fprintf(stream, "%f\n", ann_para.sample_fraction);
  fprintf(stream, "%d\n", int(ann_para.log_level));                                                                                                       
  fprintf(stream, "%s\n", ann_para.log_destination);                                                                                                      
  fprintf(stream, "%ld\n", ann_para.random_seed);                                                                                                         
} 

void loadFLANNParameters(FLANNParameters& ann_para, char* filename){ 

  FILE* fp = NULL; 

  fp = fopen(filename, "rt");                 
  if(NULL == fp){                             
    fprintf(stdout, "Fail to open file %s for reading!", filename);                                                                                        
    return;                                                                                                                                             
  }

  int tmp = 0;                                                                                                                                             
  fscanf(fp, "%d\n", &tmp);                                                                                                                                
  ann_para.algorithm = flann_algorithm_t(tmp);                                                                                                                                                                                                                                                                          
  fscanf(fp, "%d\n", &ann_para.checks);                                                                                                                   
  fscanf(fp, "%f\n", &ann_para.cb_index);                                                                                                                 
  fscanf(fp, "%d\n", &ann_para.trees);                                                                                                                    
  fscanf(fp, "%d\n", &ann_para.branching);                                                                                                                
  fscanf(fp, "%d\n", &ann_para.iterations);                                                                                                               
  tmp = 0;                                                                                                                                               
  fscanf(fp, "%d\n", &tmp);                                                                                                                               
  ann_para.centers_init = flann_centers_init_t(tmp);                                                                                                  

  fscanf(fp, "%f\n", &ann_para.target_precision);                                                                                                        
  fscanf(fp, "%f\n", &ann_para.build_weight);                                                                                                               
  fscanf(fp, "%f\n", &ann_para.memory_weight);                                                                                                           
  fscanf(fp, "%f\n", &ann_para.sample_fraction);                                                                                                 
  fclose(fp);  
}

size_t min(size_t a, size_t b){
  return a>b?b:a;
};

size_t max(size_t a, size_t b){
  return a>b?a:b;
};
