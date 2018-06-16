void FIND_EIGEN_PROD (float * , float2 *, float * , cufftHandle , float * , float * , float2 * , cufftHandle , float * , float *);

void FIND_ZB    (float * , float * , float * , float *);

void FIND_GRAD(float * , float *, float *, float *, float *, float *, float *, float *, float *, float *, double *, double *, double *, double *, float *, float *, float2 *, float *, cufftHandle, float *, float *, float *, float *, float *, float *, double *, double *, float *);

void FIND_LIK(float * , float * , float * , float *, float * , float * , float * , double * , double * , float * , float * , float * , float * , float * , double * , int * );

void SAVE_ORIGINALS(double * , float * , float * , float * , float * , float * , float * , float * , float * , float * , float * , float * , float * , float * , float *, float *, float *, float *, float *, float *, float *);

void GENERATE_MOMENTUM(float * , float * , float *,curandGenerator_t, float * , float * , float * , float * , float *, float *, float *, float *);

void UPDATE_MOMENTUM(float , float * , float * , float * , float * , float * , float * , float * , float * , float *);

void UPDATE_PARAMS(float , float * , float * , float * , float * , float * , float * , float* , float *, float *, float *, float *, float *, float *);

void FIND_KINETIC(float * , float * , float * , double * , double * , double * , float * , float * , float * , float *, float * , float * , float *);

void REVERT_STATE(float * , float * , float * , float * , float * , float * , float * , float * , float * , float * , float * , float * , float * , float *, float *, float *, float *, float *, float *, float *);

void FIND_KINETIC_GAMMA(float * , double * , double * , double *);

void SNAPSHOT(float * , float * , float * , float * , float * , float * , float * , float *);

void UPDATE_RFX(double * , double * , float * , float *, int * , float * , float * , float * , float * , unsigned long * , int *);

