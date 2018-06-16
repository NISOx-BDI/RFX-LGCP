// The following two are related to the mask we use
__const__ int V=3981312; 
__const__ float sqV = 1995.323f;


const int NB = 32;  // Blocks per grid  
const int NT = 256; // Threads per block



__global__ void DragRealBig(float2* , float* , float* );

__global__ void DragRealGrad(float2 * , float * , float *);

__global__ void DragEigenBig (float2 * , float *);

__global__ void FindBaseBig(float * , float2 * , float *);

__global__ void LoadVec(float * , float2 *);

__global__ void LoadAddVec(float2 * , float2 *);

__global__ void LoadAddVecSecond( float * , float2 * );

__global__ void SendToCovariates(float* , float*);

__global__ void SaveRFX (double* , float* );

__global__ void GradFirst(float * , float * , float * , float * , float2 * , float * , float *);

__global__ void GradSecond(float * , float * , float * , float * , float * , float * , float *);

__global__ void LikFirst(float* , float * , float * , float * , float * , float * , float * );

__global__ void LikSecond(float * , double *);

__global__ void Replace(float * , float *);

__global__ void KineticFirst(float * , double *);

__global__ void Update(float * , float * , float );

__global__ void UpdateScalars(float * , float * , float , float * );

__global__ void UpdateSecond(float * , float * , float , float*);

__global__ void SetElement(float* , int , float );

__global__ void CrossVector(float * , float *);

__global__ void GradBeta(float * , double* );

__global__ void GradSigma(float * , double * , float * , float * );

__global__ void GradRho(float * , double * , float * , float *);

__global__ void saveGPsDiff(double* , double* , float * , float * , float *);

__global__ void saveGPs(double * , double * , float * , float * , float *);

__global__ void rfxSum(float * , float * , float * , float * , double * );

__global__ void Print(float * , float * , float * , int );



__const__ int I=157;
__const__ int K_star=4;
__const__ int K=3;
__const__ int TIMES=4;
__const__ int STUDIES=42;
__const__ int CUBLAS_TMP=1;
const double rfx_phi = 10.0;
