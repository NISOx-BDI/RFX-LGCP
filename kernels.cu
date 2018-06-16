#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cublas.h>
#include <curand.h>

#include "kernels.h"


// Save the matrix-vector product
__global__ void DragRealBig(float2 *ORIGIN , float *Cg , float *CSg) {
    
    // trace the element that you are interested in
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    // Throw Cgamma where it belongs
    Cg[idx] = ORIGIN[idx].x/sqV;
    // Throw CSgamma where it belongs
    CSg[idx] = ORIGIN[K*V+idx].x/sqV;

}





// In the gradient of \gamma, this kernel takes the values from the FFT object and stores them into the gradient vector.
__global__ void DragRealGrad(float2 *ORIGIN , float *DEST , float *VEC) {
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    DEST[idx] = ORIGIN[idx].x/sqV - VEC[idx];
}




// Saves the eigenvalues
__global__ void DragEigenBig (float2 *ORIGIN , float *DEST) {

    // Trace the element
    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    // Grab the eigenvalues
    float tmp;
    tmp = ORIGIN[idx].x;
    tmp = sqrtf(tmp); 

    // Save the eigenvalue
    ORIGIN[idx].x = tmp;
    ORIGIN[idx].y = 0.0f;
    // Also save it for the gradient calculations
    DEST[idx] = tmp;

    // And finally divide the bottom part with the upper part. See formulas in paper for details.
    ORIGIN[K*V+idx].x /=tmp;
    ORIGIN[K*V+idx].y = 0.0f;

}


// This kernel will calculate the base of the two Nested Block Circulant matrices, given the correlation decay parameter.
__global__ void FindBaseBig (float *rho , float2 *DFT , float *dist) {

    int idx = threadIdx.x + blockIdx.x*blockDim.x;  // Element identifier   ( 0 - K*V-1 )
    int idy = floorf( (float)idx/(float)V );        // Covariate identifier ( 0 - K-1   )
    int idz = idx-idy*V;
    float tmp1 , tmp2 ;
    tmp1 = powf( dist[idz],1.9f );
    tmp2 = expf( -rho[idy]*tmp1/100.0f );

    // Base of the first matrix
    DFT[idx].x = tmp2;
    DFT[idx].y = 0.0f;

    //Base of the second matrix
    DFT[K*V+idx].x = tmp1*tmp2;
    DFT[K*V+idx].y = 0.0f;
}



// For a given vector, this kernel throws it to an FFT object so the DFT/IDFT can be found.
__global__ void LoadVec(float *vector , float2 *FFT) {
    int idx = threadIdx.x + blockIdx.x*blockDim.x; // this should span the full range of the vector
    FFT[idx].x = vector[idx]; // The real part is replaced by the vector value
    FFT[idx].y = 0.0f;        // The imaginary part is zero. The following kernel also replaces the imaginary part
}



// The following kernel takes the values of the IDFT(\gamma) and puts them into the big FFT object for the two multiplications. 
__global__ void LoadAddVec(float2 *FFT_small , float2 *FFT_big) {

    int idx = threadIdx.x + blockIdx.x*blockDim.x; // this should span the full range of the vector
    float tmp1,tmp2;
    tmp1 = FFT_small[idx].x; // the real part of the IDFT(\gamma)          
    tmp2 = FFT_small[idx].y; // the imaginary part

    // First fill the upper part of the big DFT object.
    FFT_big[idx].y  = FFT_big[idx].x*tmp2/sqV;
    FFT_big[idx].x *=  tmp1/sqV;
    

    // And then fill the bottom part of the big DFT object.
    FFT_big[K*V+idx].y  =  FFT_big[K*V+idx].x*tmp2/sqV;
    FFT_big[K*V+idx].x *=  tmp1/sqV;
    
}



// This is similar to the kernel above. It's going the other direction around to put the elements together
__global__ void LoadAddVecSecond(float *vector , float2 *FFT) {

    int idx = threadIdx.x + blockIdx.x*blockDim.x; // this should span the full range of the vector
    FFT[idx].x *= vector[idx]/sqV;
    FFT[idx].y *= vector[idx]/sqV;          
}


/* Save the exp(Z\beta) product for use in gradient/likelihood kernels */
__global__ void SendToCovariates(float *ZB , float *COVARIATES){

    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if (idx<I){
        COVARIATES[idx*(K_star+2)] = expf(ZB[idx]);    /* CHANGE RFX */
    }

}


/* Save the new intensity random effects for use in gradient/likelihood kernels */ /* CHANGE RFX */
__global__ void SaveRFX (double* rfx, float *COVARIATES) {

    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if (idx<I){
        COVARIATES[ idx*(K_star+2)+1+K_star ] = (float)rfx[idx];
    }
}


/* GRADIENT KERNELS */


// FInds the per-voxel constant terms
__global__ void GradFirst(float *voxel_tmp , float *sigma , float *Cgamma , float *vol , float2 *DFT_grad , float *COVARIATES , float *TERM_GAMMA) {

    // define the variables  /* RFX CHANGE :changed the entire kernel basically */
    int tidx = threadIdx.x;                          // which thread of the block we are at
    int idx = threadIdx.x + blockIdx.x*blockDim.x;   // which voxel we are at
    __shared__ float CACHE[NT];
    int i;
    int si;          // this counts which study we are at.
    int ci,icache;   // counter for the covariates
    int mn;
    int mx;
    float c;                 // this is the exponential term that always appears
    float sum[K_star];
    float error[K_star];
    float t1,t2;            // To correct for floating point operations
    for (i=0 ; i<K_star ; i++) {
        sum[i] = 0.0f;
        error[i] = 0.0f;
    }

    // first set the variables equal to zero and add some other terms that need to be added.
    for (ci=0 ; ci<K ; ci++){
        voxel_tmp[ci+K_star*idx] = 0.0f;
        DFT_grad[idx+ci*V].x = -TERM_GAMMA[idx+ci*V];
        DFT_grad[idx+ci*V].y = 0.0f;
    }
    for (ci=K ; ci<K_star ; ci++){
        voxel_tmp[ci+K_star*idx] = 0.0f;
    }


    for (i=0 ; i<TIMES ; i++) {
        mn = i*STUDIES;                      // which study you start at
        mx = mn+STUDIES;                     // which study you finish at
        mx = (mx<=I)*mx + (mx>I)*I;          // have to take into cinsideration that at the end there is something that is not used at all
        __syncthreads();
        CACHE[tidx] = COVARIATES[tidx+mn*(K_star+2)];         // this loads the shared data 
        __syncthreads();
        // and now perform the loop over studies of the same chunk for this voxel
        for (si=mn ; si<mx ; si++){
            icache = (si-mn)*(K_star+2);                      // Where to start looking at
            c = 0.0f;
            for (ci=0 ; ci<K ; ci++){
                c += sigma[ci]*Cgamma[idx+ci*V]*CACHE[icache+ci+1]; // +1 since position 0 is reserved for exp(Z\beta)
            }
            c = expf(c)*vol[idx]*CACHE[icache]*CACHE[icache+1+K_star];  // Main input for voxel-summation terms
        
            /* Add this to running sum of studies */
            for (ci=0 ; ci<K_star ; ci++) {
                // Correct for floating point arithmetic
                t1 = c*CACHE[icache+ci+1]-error[ci];
                t2 = sum[ci]+t1;
                error[ci] = (t2-sum[ci])-t1;
                sum[ci] = t2;
            }
        }
    }
    

    // the study running sum has been found. now just send it where it belongs in the temporary variable
    for (ci=0 ; ci<K ; ci++) {
        voxel_tmp[ci+K_star*idx] += sum[ci];
        DFT_grad[idx+ci*V].x += sum[ci];
    }
    for (ci=K ; ci<K_star ; ci++) {
        voxel_tmp[ci+K_star*idx] += sum[ci];
    }

    // and now simply multiply what is inside the fft object by sigma
    for (ci=0 ; ci<K ; ci++){
        DFT_grad[idx+ci*V].x = -sigma[ci]*DFT_grad[idx+ci*V].x;
    }

}



/* LIKELIHOOD KERNELS */



// Per voxel likelihood contributions are found by this kernel
__global__ void LikFirst(float *lik_tmp , float *gamma , float *COVARIATES , float *sigma , float *Cgamma , float *TERM_GAMMA , float *vol) {

    int tidx = threadIdx.x;                             // which thread of the block we are at
    int idx = threadIdx.x + blockIdx.x*blockDim.x;      // which voxel we are at
    __shared__ float CACHE[NT];                         // this will hold the covariates
    int i;
    int si; // this counts which study we are at.
    int ci,icache; // counter for the covariates
    int mn;
    int mx;
    float sum=0;
    float error=0;
    float t1,t2;
    float c;

    // first add the prior contributions of the gammas
    lik_tmp[idx] = 0.0f;
    for (ci=0 ; ci<K ; ci++) {
        lik_tmp[idx] += gamma[idx+ci*V]*gamma[idx+ci*V];
    }
    lik_tmp[idx] *= -0.5f;



    for (i=0 ; i<TIMES ; i++) {
        mn = i*STUDIES;                      // which study you start at
        mx = mn+STUDIES;                     // which study you finish at
        mx = (mx<=I)*mx + (mx>I)*I;          // have to take into cinsideration that at the end there is something that is not used at all
        __syncthreads();
        CACHE[tidx] = COVARIATES[tidx+mn*(K_star+2)];         // this loads the shared data
        __syncthreads();
        // now that some part of the data is loaded, do calculations for this part
        for (si=mn ; si<mx ; si++){
            icache = (si-mn)*(K_star+2);
            c = 0.0f; // set to zero
            for (ci=0 ; ci<K ; ci++){
                c += sigma[ci]*Cgamma[idx+ci*V]*CACHE[icache+ci+1]; // were are adding one because the first position is reserved for the exponential term.
            }
            c = expf(c)*vol[idx]*CACHE[icache]*CACHE[icache+1+K_star];
            t1 = c-error;
            t2 = sum+t1;
            error = (t2-sum)-t1;
            sum = t2;
        }
    }

    // and now simply add it to the corresponding place in the tmp variable
    // we can also add the TERM_GAMMA term at the same time
    for (ci=0 ; ci<K ; ci++) {
        t1 = (-TERM_GAMMA[idx+ci*V]*Cgamma[idx+ci*V]*sigma[ci])-error;
        t2 = sum+t1;
        error = (t2-sum)-t1;
        sum = t2;
    }

    lik_tmp[idx] -= sum; // this is where the minus sign is taken care of
    
}



// This kernel sums up sets of voxel terms. The final summation is found in the CPU 
__global__ void LikSecond(float *lik_tmp , double *lik_part) {

    __shared__ double likes[NT];
    int tidx = threadIdx.x;                        // which thread of the block we are at
    int idx = threadIdx.x + blockIdx.x*blockDim.x; // which voxel we are at
    double sum = 0.0;

    while(idx<V) {
        sum += lik_tmp[idx];
        idx += blockDim.x * gridDim.x;
    }
    likes[tidx] = sum;
    __syncthreads();
	
	int bidx = (blockDim.x>>1);
    while ( bidx!=0 ){
        if (tidx < bidx) {
            likes[tidx] += likes[tidx+bidx];
        }
        __syncthreads();
        bidx = bidx>>1;
    }

    if (tidx == 0){
        // printf("\n 0 is %.4f ",likes[0]);   	 
        lik_part[blockIdx.x] = likes[0];
    }

}










// Replaces the some value with some other value in a vector 
__global__ void Replace(float *WHAT , float *WHERE) 
{

    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    WHERE[idx] = WHAT[idx];

}




// Helps calculating kinetic energy
__global__ void KineticFirst(float *mom_gamma , double *kin_part)
{
    __shared__ double kins[NT];
    int tidx = threadIdx.x;                          // which thread
    int idx = threadIdx.x + blockIdx.x * blockDim.x; // which voxel
    int ci;
    double sum = 0; 
	
    while (idx < V) {
		for (ci=0 ; ci<K ; ci++) {
            sum += (double) powf(mom_gamma[idx+ci*V],2.0f);
        }
        idx += blockDim.x * gridDim.x;
    }
	
	kins[tidx] = sum;
	
	__syncthreads();
	
	int i = (blockDim.x >> 1);
	while (i != 0) {
		if (tidx < i)
			kins[tidx] += kins[tidx + i];
		__syncthreads();
		i = i >> 1;	
	}
	
	if (tidx == 0)
		kin_part[blockIdx.x] = kins[0];
}



// Adds the increments in the HMC
__global__ void Update(float *WHAT , float *WITH , float AMOUNT) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x; // which voxel
    WHAT[idx] +=AMOUNT*WITH[idx];
}


// Same as above for the scalar parameters
__global__ void UpdateScalars(float *WHAT , float *WITH , float AMOUNT , float *MASS) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x; // this defines the element
    WHAT[idx] += AMOUNT*WITH[idx]/MASS[idx];
}


// For HMC with boundary constraints
__global__ void UpdateSecond(float *WHAT , float *WITH , float AMOUNT , float *MULT)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x; 
    WHAT[idx] *=MULT[idx];
    WHAT[idx] +=AMOUNT*WITH[idx];
    MULT[idx] = 1.0f;
}



// This kernel replaces a given position of a vector with a number
__global__ void SetElement(float *vector , int position , float what) {
    vector[position] = what;
}



// Cross product of first with sqrt(second). Used to scale the momentum according to the desired variance
__global__ void CrossVector(float *first , float *second) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x; // the element of the vector
    first[idx] *= sqrtf(second[idx]);

}


/* The following kernels find partial sums for the gradient */

__global__ void GradSigma(float *voxel_tmp , double *sigma_part , float *Cgamma , float *TERM_GAMMA ) {

    // First allocate some shared memory to store the block partial sums.
    __shared__ double block_part[K*NT];
    int tidx = threadIdx.x; // which thread of the block we are at
    int idx = threadIdx.x + blockIdx.x*blockDim.x; // which voxel we are at
    int ci;
    
    // Define the variables that hold the partial sums
    double sum[K];
    for (ci=0 ; ci<K ; ci++){
        sum[ci] = 0;
    }

    // start looping over voxels
    while(idx<V) {
        for (ci=0 ; ci<K ; ci++){
            sum[ci] += Cgamma [idx+ci*V]*(voxel_tmp[ci+K_star*idx]-TERM_GAMMA[idx+ci*V]);
        }
        idx += blockDim.x*gridDim.x;
    }

    // Now store those partial sums into the shared memory
    for (ci=0 ; ci<K ; ci ++) {
        block_part[tidx+ci*NT]             = sum[ci]; 
    }

    __syncthreads();


    int bidx = (blockDim.x>>1);
    while ( bidx!=0 ){
        if (tidx < bidx) {
            for (ci=0 ; ci<K ; ci++){
                block_part[tidx+ci*NT] += block_part[tidx+bidx+ci*NT];
            }
        }
        __syncthreads();
        bidx = bidx>>1;
    }

    // Save the partial sums only for the first thread of a block
    if (tidx==0) {
        // Save the partial sum for \sigma
        for (ci=0 ; ci<K ; ci++) {
            sigma_part[ci+K*blockIdx.x] = block_part[ci*NT];
        }
    }

}


__global__ void GradBeta(float *voxel_tmp , double *beta_part) {

    // First allocate some shared memory to store the block partial sums.
    __shared__ double block_part[K_star*NT];
    int tidx = threadIdx.x; // which thread of the block we are at
    int idx = threadIdx.x + blockIdx.x*blockDim.x; // which voxel we are at
    int ci;
    
    // Define the variables that hold the partial sums
    double sum[K_star];
    for (ci=0 ; ci<K_star ; ci++){
        sum[ci] = 0;
    }

    // start looping over voxels
    while(idx<V) {
        for (ci=0 ; ci<K_star ; ci++){
            sum[ci] += voxel_tmp[ci+K_star*idx];
        }
        idx += blockDim.x*gridDim.x;
    }

    // Now store those partial sums into the shared memory
    for (ci=0 ; ci<K_star ; ci ++) {
        block_part[tidx+ci*NT]             = sum[ci]; 
    }

    __syncthreads();


    int bidx = (blockDim.x>>1);
    while ( bidx!=0 ){
        if (tidx < bidx) {
            for (ci=0 ; ci<K_star ; ci++){
                block_part[tidx+ci*NT] += block_part[tidx+bidx+ci*NT];
            }
        }
        __syncthreads();
        bidx = bidx>>1;
    }

    // Save the partial sums only for the first thread of a block
    if (tidx==0) {
        // Save the partial sum for \sigma
        for (ci=0 ; ci<K_star ; ci++) {
            beta_part[ci+K_star*blockIdx.x] = block_part[ci*NT];
        }
    }

}



__global__ void GradRho(float *voxel_tmp , double *rho_part , float *CSgamma , float *TERM_GAMMA ) {

    // First allocate some shared memory to store the block partial sums.
    __shared__ double block_part[K*NT];
    int tidx = threadIdx.x; // which thread of the block we are at
    int idx = threadIdx.x + blockIdx.x*blockDim.x; // which voxel we are at
    int ci;
    
    // Define the variables that hold the partial sums
    double sum[K];
    for (ci=0 ; ci<K ; ci++){
        sum[ci] = 0;
    }

    // start looping over voxels
    while(idx<V) {
        for (ci=0 ; ci<K ; ci++){
            sum[ci] += CSgamma [idx+ci*V]*(voxel_tmp[ci+K_star*idx]-TERM_GAMMA[idx+ci*V]);
        }
        idx += blockDim.x*gridDim.x;
    }

    // Now store those partial sums into the shared memory
    for (ci=0 ; ci<K ; ci ++) {
        block_part[tidx+ci*NT]             = sum[ci]; 
    }

    __syncthreads();


    int bidx = (blockDim.x>>1);
    while ( bidx!=0 ){
        if (tidx < bidx) {
            for (ci=0 ; ci<K ; ci++){
                block_part[tidx+ci*NT] += block_part[tidx+bidx+ci*NT];
            }
        }
        __syncthreads();
        bidx = bidx>>1;
    }

    // Save the partial sums only for the first thread of a block
    if (tidx==0) {
        // Save the partial sum for \sigma
        for (ci=0 ; ci<K ; ci++) {
            rho_part[ci+K*blockIdx.x] = block_part[ci*NT];
        }
    }
}



/* Save the running sums and sums of square of the latent gps */
__global__ void saveGPsDiff(double *gp_s , double *gp_ss , float *Cgamma , float *beta , float *sigma) {

    int idx = threadIdx.x + blockIdx.x*blockDim.x; // voxel location
    int column = 0;
    int i,j;
    float tmp[K];

    /* Coefficients */
    for (i=0 ; i<K ; i++) {
        tmp[i] = beta[i] + sigma[i]*Cgamma[i*V + idx];
        gp_s[column*V + idx]  += (double)tmp[i];
        gp_ss[column*V + idx] += (double)powf(tmp[i],2.0f);
        column = column+1;
    }

    /* Differences */
    for (i=0 ; i<K ; i++) {
        for (j=(i+1) ; j<K ; j++ ) {
            gp_s[column*V + idx]  += (double)( tmp[i] - tmp[j] );
            gp_ss[column*V + idx] += (double)powf( tmp[i] - tmp[j] ,2.0f); 
            column = column+1;
        }
    }
}

/* Same as above when there is only intercept is spatially varying */
__global__ void saveGPs(double *gp_s , double *gp_ss , float *Cgamma , float *beta , float *sigma) {

    int idx = threadIdx.x + blockIdx.x*blockDim.x; // voxel location 
    int i;
    float tmp;

    for (i=0 ; i<K ; i++) {
        tmp = beta[i] + sigma[i]*Cgamma[i*V + idx];
        gp_s[i*V + idx]  += (double)tmp;
        gp_ss[i*V + idx] += (double)powf(tmp,2.0f);
    }

}



/* To find the sum over voxels for random effects */
__global__ void rfxSum(float *sigma , float *Cgamma , float *vol , float *COVARIATES , double *rfx )
{
    int idx = threadIdx.x + blockIdx.x*blockDim.x;     /* study ID  */
    double sum=0.0;                                    /* the sum over voxels for each study */ 
    double tmp=0.0;
    int i; int vi;
    float covars[K];                                   /* spatially varying study covariates */ 

    
    if (idx<I){
        /* get the spatially varying covariates of each study */
        for (i=0 ; i<K ; i++) {
            covars[i] = COVARIATES[idx*(K_star+2)+1+i];
        }  
        /* add up the contributions from all voxels */         
        for (vi=0 ; vi<V ; vi++) {
            if (vol[vi]>0.0f){
                tmp = 0.0;
                for (i=0 ; i<K ; i++) {
                    tmp += (double) (sigma[i]*Cgamma[i*V + vi]*covars[i]);
                }
                sum += exp(tmp);
            }   
        }
        /* Save the sum to the random effects */
        rfx[idx] = sum; 
    }
       
}


__global__ void Print(float *beta, float *sigma, float *rho, int iter ) 
{
    printf("\n %d -- 1) b %.5f -- s %.5f -- r %.5f ",iter,beta[0],sigma[0],rho[0]);
    printf("\n %d -- 2) b %.5f -- s %.5f -- r %.5f ",iter,beta[1],sigma[1],rho[1]);
    printf("\n %d -- 3) b %.5f -- s %.5f -- r %.5f ",iter,beta[2],sigma[2],rho[2]);
    

}



