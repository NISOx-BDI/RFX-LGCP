#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cublas.h>
#include <curand.h>


#include "kernels.h"
#include "randgen.h"

// some CPU variables
extern float Htau;
extern int HI;
extern int HK_star;
extern int HK;
extern int V_extended;


const float RHO_UPPER = 2.00f;
const float RHO_LOWER = 0.35f;
const double VOXEL_VOL = 8.0;


// Finds the matrix-vector products
void FIND_EIGEN_PROD (float *rho , float2 *DFT_big , float *dist , cufftHandle plan_big , float *eigen , float *gamma , float2 *DFT , cufftHandle plan , float *Cgamma , float *CSgamma) {

    // 2 nested block circulant matrices appear in the calculations. The first kernel finds the bases.
    FindBaseBig<<<K*V/NT,NT>>>(rho,DFT_big,dist);
    // Load the \gamma vector in the small DFT
    LoadVec<<<K*V/NT,NT>>>(gamma,DFT);
    cudaDeviceSynchronize();
    // Execute the forward DFT
    cufftExecC2C(plan_big,DFT_big,DFT_big,CUFFT_FORWARD);
    cudaDeviceSynchronize();
    // Execute the IDFT of the vector
    cufftExecC2C(plan,DFT,DFT,CUFFT_INVERSE);

    // Save the eigenvalues of the correlation matrix. Be careful. These values are multiplied by sqrt(n) already and the square root is taken
    // At the same time prepare the big FFT object for the IDFT(gamma) that is coming. See kernel carefully.
    DragEigenBig<<<K*V/NT,NT>>>(DFT_big,eigen);
    cudaDeviceSynchronize();

    // Throw IDFT(\gamma) into the big object
    LoadAddVec<<<K*V/NT,NT>>>(DFT,DFT_big);
    cudaDeviceSynchronize();

    // Perform the final DFT
    cufftExecC2C(plan_big,DFT_big,DFT_big,CUFFT_FORWARD);
    cudaDeviceSynchronize();

    // And finally save the products
    DragRealBig<<<K*V/NT,NT>>>(DFT_big,Cgamma,CSgamma);
    cudaDeviceSynchronize();

}


// Finds the design matrix times coefficients product
void FIND_ZB(float *Z , float * B , float * ZB , float *COVARIATES) {

    // Calculate the product
    cublasSgemv('n',HI,HK_star,1.0f,Z,HI,B,1,0.0f,ZB,1);
    cudaDeviceSynchronize();
    // Then save to a vector
    SendToCovariates<<<CUBLAS_TMP,512>>>(ZB,COVARIATES);
    cudaDeviceSynchronize();
    
}






// The following function calculates the gradient.
// For grad_gamma the minus sign is take care of when multiplying by sigma
// For grad_beta and grad_sigma the minus sign is taken care of when the partial sums are summed in the CPU
void FIND_GRAD(float *gamma , float *grad_gamma , float *beta , float *grad_beta , float *sigma , float *grad_sigma , float *voxel_tmp , float *ZB , float *Cgamma , float *vol , double *sigma_part , double *beta_part , double *Hsigma_part , double *Hbeta_part , float * COVARIATES , float *TERM_GAMMA , float2 *DFT , float *eigen , cufftHandle plan , float *Hsigma , float *Hbeta , float *HZN , float *Hrho , float *rho , float *CSgamma , double *rho_part , double *Hrho_part , float *grad_rho) {
    
    // The first kernel calculates common term for all the variables per voxel. It stores it in voxel_tmp.
    GradFirst<<<V/NT,NT>>>(voxel_tmp ,sigma ,Cgamma ,vol , DFT , COVARIATES,TERM_GAMMA);
    cudaDeviceSynchronize();

    // The lines below calculate the FFT product that appears in gradient calculations
    // The vector is already loaded in the FFT object so we just find its IDFT
    cufftExecC2C(plan, DFT, DFT, CUFFT_INVERSE);
    cudaDeviceSynchronize();

    // Now load the eigenvalues of the square root correlation matrix
    LoadAddVecSecond<<<K*V/NT,NT>>>(eigen,DFT);
    cudaDeviceSynchronize();

    // And finally find the matrix vector product
    cufftExecC2C(plan, DFT, DFT, CUFFT_FORWARD);
    cudaDeviceSynchronize();

    // To finish, send the values to the gradient of \gamma
    DragRealGrad<<<K*V/NT,NT>>>(DFT,grad_gamma,gamma);
    cudaDeviceSynchronize();

    // This  series of kernels adds up the partial sums for the gradients of scalar parameters
    GradSigma<<<NB,NT>>>(voxel_tmp , sigma_part , Cgamma , TERM_GAMMA );
    GradBeta<<<NB,NT>>>(voxel_tmp , beta_part);
    GradRho<<<NB,NT>>>(voxel_tmp,rho_part,CSgamma,TERM_GAMMA);
    cudaDeviceSynchronize();

    // Move the partial sums and the parameters to the CPU for the adding up
    int s,ss;
    cudaMemcpy(Hsigma,sigma, HK*sizeof(float),      cudaMemcpyDeviceToHost);
    cudaMemcpy(Hrho,  rho,   HK*sizeof(float),      cudaMemcpyDeviceToHost);
    cudaMemcpy(Hbeta, beta,  HK_star*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Hsigma_part, sigma_part, NB*HK*sizeof(double),      cudaMemcpyDeviceToHost);
    cudaMemcpy(Hrho_part,   rho_part,   NB*HK*sizeof(double),      cudaMemcpyDeviceToHost);
    cudaMemcpy(Hbeta_part,  beta_part,  NB*HK_star*sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Dont get confused. I will find the derivatives inside the parameter vectors.
    for (s=0 ; s<HK ; s++) {
        Hrho[s] = Hsigma[s];
        Hsigma[s] *= -Htau;
    }
    for (s=0 ; s<HK_star ; s++) {
        Hbeta[s] *= -Htau;
        Hbeta[s] += HZN[s];
    }

    // First add the partial sums found earlier in the GPU
    double tmp_sigma[HK];
    double tmp_rho[HK];
    double tmp_beta[HK_star];
    for (ss=0 ; ss<NB ; ss++){
        for (s=0 ; s<HK ; s++) {
            tmp_sigma[s] += Hsigma_part[s+HK*ss];
            tmp_rho[s]   += Hrho_part[s+HK*ss];
        }
        for (s=0 ; s<HK_star ; s++) {
            tmp_beta[s] += Hbeta_part[s+HK_star*ss];
        }
    }

    // And then add the sums to the gradients of the parameters
    for (s=0 ; s<HK ; s++) {
            Hsigma[s] += -(float)tmp_sigma[s];
            Hrho[s]   *=  (float)tmp_rho[s]/200; // The 200 appears due to the parametrisation
        }
    for (s=0 ; s<HK_star ; s++) {
            Hbeta[s]  += -(float)tmp_beta[s];
    }


    // And finally send everything back to the GPU
    cudaMemcpy(grad_sigma, Hsigma, HK*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(grad_rho,   Hrho,   HK*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(grad_beta,  Hbeta,  HK_star*sizeof(float),cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

}







// Finds the log-likelihood
void FIND_LIK(float *lik_tmp , float *gamma , float *COVARIATES , float * sigma , float *Cgamma , float *TERM_GAMMA , float *vol , double *lik_part , double *Hlik_part , float *ZB , float *HZB , float *Hsigma , float *beta , float *Hbeta , double *likelihood , int *Hcounts )
{
    // First do the two kernels required in the GPU
    LikFirst<<<V/NT,NT>>>(lik_tmp,gamma,COVARIATES,sigma,Cgamma,TERM_GAMMA,vol);
    cudaDeviceSynchronize();
    LikSecond<<<NB,NT>>>(lik_tmp,lik_part);
    cudaDeviceSynchronize();

    // Define two variables that will be used
    int s;
    double sum=0;

    // Transfer everything you need to the CPU
    cudaMemcpy(Hlik_part, lik_part, NB*sizeof(double),      cudaMemcpyDeviceToHost);
    cudaMemcpy(HZB,       ZB,       HI*sizeof(float),      cudaMemcpyDeviceToHost);
    cudaMemcpy(Hsigma,    sigma,    HK*sizeof(float),      cudaMemcpyDeviceToHost);
    cudaMemcpy(Hbeta,     beta,     HK_star*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Add the likelihood partial sums together
    for (s=0 ; s<NB ; s++) {
        sum += Hlik_part[s];
    }

    // Add the annoying term that appears in the likelihood equation.
    for (s=0 ; s<HI ; s++) {
        sum += (double) HZB[s]*Hcounts[s];
    }

    // Finally add the prior contributions
    for (s=0 ; s<HK ; s++){
        sum += -0.5*Htau*Hsigma[s]*Hsigma[s];
    }
    for (s=0 ; s<HK_star ; s++){
        sum += -0.5*Htau*Hbeta[s]*Hbeta[s];
    }

    // Now just save the value that you obtained.
    likelihood[0] = sum;

}




// If a move of the HMC is accepted then save the new values
void SAVE_ORIGINALS(double *likelihood , float *beta , float *ori_beta , float *sigma , float *ori_sigma , float *gamma , float *ori_gamma , float *Cgamma , float *ori_Cgamma , float *grad_beta , float * ori_grad_beta , float *grad_sigma , float *ori_grad_sigma , float *grad_gamma , float *ori_grad_gamma , float *rho, float *ori_rho, float *grad_rho, float *ori_grad_rho, float *CSgamma, float *ori_CSgamma)
{
    // Replace the proposed values with the original
    likelihood[1] = likelihood[0];
    
    Replace<<<1,K_star>>>(beta,ori_beta);
    Replace<<<1,K>>>(rho,ori_rho);
    Replace<<<1,K>>>(sigma,ori_sigma);
    
    Replace<<<K*V/NT,NT>>>(gamma,ori_gamma);
    Replace<<<K*V/NT,NT>>>(Cgamma,ori_Cgamma);
    Replace<<<K*V/NT,NT>>>(CSgamma,ori_CSgamma);
    
    Replace<<<1,K_star>>>(grad_beta,ori_grad_beta);
    Replace<<<1,K>>>(grad_sigma,ori_grad_sigma);
    Replace<<<K*V/NT,NT>>>(grad_gamma,ori_grad_gamma);
    Replace<<<1,K>>>(grad_rho,ori_grad_rho);
    
    
    cudaDeviceSynchronize();
}



// This function generates the momentum. CUDA won't generate odd number of variables hence the if statement. 
void GENERATE_MOMENTUM(float *mom_beta , float *mom_sigma , float *mom_gamma,curandGenerator_t gen , float *mom_beta_even , float *mom_sigma_even , float *mom_gamma_even , float *Mass_sigma , float *Mass_beta, float *mom_rho, float *mom_rho_even, float *Mass_rho)
{
    // Generate the N(0,1) momentum for the variables of interest
    // Overall means
    if ( (HK_star%2)==0 ) {
        curandGenerateNormal(gen , mom_beta , K_star , 0.0f , 1.0f);
    } else {
        curandGenerateNormal(gen, mom_beta_even,  K_star+1 , 0.0f, 1.0f);
        Replace<<<1,HK_star>>>(mom_beta_even,mom_beta);
    }
    // Marginal standard deviations
    if ( (HK%2)==0 ) {
        curandGenerateNormal(gen, mom_sigma, HK   , 0.0f, 1.0f);
        curandGenerateNormal(gen, mom_rho,  HK   , 0.0f, 1.0f);
    } else {
        curandGenerateNormal(gen, mom_sigma_even,  HK+1 , 0.0f, 1.0f);
        Replace<<<1,HK>>>(mom_sigma_even,mom_sigma);
        curandGenerateNormal(gen, mom_rho_even,    HK+1 , 0.0f, 1.0f);
        Replace<<<1,HK>>>(mom_rho_even,mom_rho);
    }
    // gamma vactors
    curandGenerateNormal(gen, mom_gamma, HK*V    , 0.0f, 1.0f);
    cudaDeviceSynchronize();

    // Now scale the scalars by their masses
    CrossVector<<<1,HK_star>>>(mom_beta,Mass_beta);
    CrossVector<<<1,HK>>>(mom_sigma,Mass_sigma);
    CrossVector<<<1,HK>>>(mom_rho,  Mass_rho);
    cudaDeviceSynchronize();
}





// Finds the kinetic energy
void FIND_KINETIC(float *mom_beta , float *mom_sigma , float *mom_gamma , double *Kinetic , double *kin_part , double *Hkin_part , float *Hmom_beta , float *Hmom_sigma , float *HMass_sigma , float *HMass_beta , float *mom_rho , float *Hmom_rho , float *HMass_rho)
{
    // Execute the kernels and transfer things to the CPU
    KineticFirst<<<NB,NT>>>(mom_gamma,kin_part);
    cudaDeviceSynchronize();
    cudaMemcpy(Hkin_part,kin_part,NB*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(Hmom_beta,mom_beta,HK_star*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(Hmom_sigma, mom_sigma, HK*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(Hmom_rho  , mom_rho  , HK*sizeof(float),cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Define the variables that will be used
    int s;
    double sum = 0;

    // Add the Kinetic parts together
    for (s=0 ; s<NB ; s++) {
        sum += Hkin_part[s]; 
    }
    for (s=0 ; s<HK_star ; s++){
        sum += (Hmom_beta[s]*Hmom_beta[s])/HMass_beta[s];
    }

    for (s=0 ; s<HK ; s++){
        sum += (Hmom_sigma[s]*Hmom_sigma[s])/HMass_sigma[s]; 
    }

    for (s=0 ; s<HK ; s++){
        sum += (Hmom_rho[s]*Hmom_rho[s])/HMass_rho[s]; 
    }

    // Multiply by half
    Kinetic[0] = 0.5*sum;

}




// Updates the momentum vector according to HMC
void UPDATE_MOMENTUM(float size , float *mom_beta , float *grad_beta , float *mom_sigma , float *grad_sigma , float *mom_gamma , float * grad_gamma , float *mom_rho , float *grad_rho , float *Sign_rho)
{

    Update<<<1,HK_star>>>(mom_beta,grad_beta,size);
    Update<<<1,HK>>>(mom_sigma, grad_sigma, size);
    UpdateSecond<<<1,HK>>>(mom_rho,grad_rho,size,Sign_rho);
    Update<<<HK*V/NT,NT>>>(mom_gamma,grad_gamma,size);
    cudaDeviceSynchronize();

}

// Updates the parameter vector according to HMC
void UPDATE_PARAMS(float size , float *beta , float *mom_beta , float *sigma , float *mom_sigma , float *gamma , float *mom_gamma , float *Mass_sigma , float *Mass_beta , float *rho, float *Hrho , float *mom_rho ,  float *Mass_rho , float *Sign_rho)
{
    // Update the parameters according to the Leapfrog scheme. Remember the Mass vectors contain standard deviations
    UpdateScalars<<<1,HK_star>>>(beta,mom_beta,size,Mass_beta);
    UpdateScalars<<<1,HK>>>(sigma,mom_sigma,size,Mass_sigma);
    UpdateScalars<<<1,HK>>>(rho,mom_rho,size,Mass_rho);
    Update<<<HK*V/NT,NT>>>(gamma,mom_gamma,size);
    cudaDeviceSynchronize();

    // Bring the correlation parameters back to see if the boundary conditions are satisfied
    cudaMemcpy(Hrho , rho , HK*sizeof(float) , cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    int s;
    float tmp;

    for (s=0 ; s<HK ; s++) {
        // Upper bounds
        if (Hrho[s] > RHO_UPPER) {
            tmp = RHO_UPPER - (Hrho[s]-RHO_UPPER);
            SetElement<<<1,1>>>(rho,s,tmp);
            SetElement<<<1,1>>>(Sign_rho , s , -1.0f);
        }
        // Lower bounds
        if (Hrho[s] < RHO_LOWER) {
            tmp = RHO_LOWER + (RHO_LOWER-Hrho[s]);
            SetElement<<<1,1>>>(rho,s,tmp);
            SetElement<<<1,1>>>(Sign_rho,s,-1.0f);
        }
    }
    cudaDeviceSynchronize();

}





// For when a move is rejected
void REVERT_STATE(float *beta , float *ori_beta , float *sigma , float *ori_sigma , float *gamma , float *ori_gamma , float *Cgamma , float *ori_Cgamma , float *grad_beta , float * ori_grad_beta , float *grad_sigma , float *ori_grad_sigma , float *grad_gamma , float *ori_grad_gamma, float *rho, float *ori_rho, float *grad_rho, float *ori_grad_rho, float *CSgamma, float *ori_CSgamma)
{

Replace<<<1,K_star>>>(ori_beta,beta);
Replace<<<1,K>>>(ori_sigma,sigma);
Replace<<<1,K>>>(ori_rho,rho);

Replace<<<K*V/NT,NT>>>(ori_gamma,gamma);
Replace<<<K*V/NT,NT>>>(ori_Cgamma,Cgamma);
Replace<<<K*V/NT,NT>>>(ori_CSgamma,CSgamma);

Replace<<<1,K_star>>>(ori_grad_beta,grad_beta);
Replace<<<1,K>>>(ori_grad_sigma,grad_sigma);
Replace<<<1,K>>>(ori_grad_rho,grad_rho);
Replace<<<K*V/NT,NT>>>(ori_grad_gamma,grad_gamma);

cudaDeviceSynchronize();

}





// Find the kinetic energy for the first part where only the gamma vectors are updated
void FIND_KINETIC_GAMMA(float *mom_gamma , double *Kinetic , double *kin_part , double *Hkin_part)
{
    // Execute the kernels and transfer to the CPU 
    KineticFirst<<<NB,NT>>>(mom_gamma,kin_part);
    cudaDeviceSynchronize();
    cudaMemcpy(Hkin_part,kin_part,NB*sizeof(double),cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Define the variables that will be used
    int s=0;
    double sum=0;

    // Add things up
    for (s=0 ; s<NB ; s++) {
        sum += Hkin_part[s];
    }

    // Multiply by half
    Kinetic[0] = 0.5*sum;
}


// Saves a snapshot of the parameters
void SNAPSHOT(float *Hsigma , float *ori_sigma , float *Hrho , float *ori_rho , float *Hbeta , float *ori_beta , float *big , float *ori_gamma )
{   
    FILE *STARTING;
    int kk;

    /* Transfer the parameters to the CPU */
    cudaMemcpy(Hsigma,ori_sigma,HK*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(Hrho  ,ori_rho,HK*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(Hbeta,ori_beta,HK_star*sizeof(float),cudaMemcpyDeviceToHost);
    cudaMemcpy(big,ori_gamma,HK*V_extended*sizeof(float),cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    /* Write to the file */
    STARTING = fopen("./outputs/starting.txt","w");
    for (kk=0 ; kk<HK_star; kk++) {
        fprintf(STARTING,"%.10f\n",Hbeta[kk]);
    }
    for (kk=0 ; kk<HK ; kk++) {
        fprintf(STARTING,"%.10f\n",Hsigma[kk]);
    }
    for (kk=0 ; kk<HK ; kk++) {
        fprintf(STARTING,"%.10f\n",Hrho[kk]);
    }
    for (kk=0 ; kk<HK*V_extended ; kk++) {
        fprintf(STARTING,"%.10f\n",big[kk]);
    }
    fclose(STARTING);
}



/* Updates the study random effects */
void UPDATE_RFX(double *Hrfx , double *rfx, float *HZB, float *ZB, int *Hcounts, float *sigma, float *Cgamma, float *vol, float *COVARIATES, unsigned long *RNG, int* author)
{
    /* Find the sum over voxels and move to host */
    rfxSum<<<CUBLAS_TMP,512>>>(sigma , Cgamma , vol , COVARIATES , rfx );
    cudaDeviceSynchronize();
    cudaMemcpy(Hrfx,rfx,HI*sizeof(double),cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    /* Add the constant terms */
    int i;
    cudaMemcpy(HZB, ZB, HI*sizeof(float), cudaMemcpyDeviceToHost);
    for (i=0 ; i<HI ; i++) {
        Hrfx[i] *= VOXEL_VOL*exp((double)HZB[i]);
    }

    /* Sample the random effect terms from their Gamma full conditionals */
    int n_authors = author[HI-1] + 1 ;
    int author_first, author_last, flag, j;
    double shape, rate, tmp;
    for (i=0 ; i<n_authors ; i++) {
        
        /* Find the first study from the i-th author */ 
        flag=0; j=-1;
        while (flag==0){
            j += 1;
            flag = ( i == (author[j]) );
        }
        author_first = j;
        /* Find the last study from the i-th author */ 
        for (j=author_first ; j<HI ; j++) {
            if (author[j] == i){
                author_last = j;
            }
        }
        /* Find the shape and the rate of the Gamma full conditional */
        shape = rfx_phi; rate = rfx_phi;
        for (j=author_first ; j<=author_last ; j++) {
            shape += (double)Hcounts[j];
            rate  += Hrfx[j];
        } 
        /* Draw the new random effect */
        tmp = rgamma(shape,rate,RNG);
        /* Save the draw */
        for (j=author_first ; j<=author_last ; j++) {
            Hrfx[j] = tmp;
        }
        /* Print something to make sure */
        if (i==100) {
            printf("\nStudy %d First %d Last %d Shape %.5f Rate %.5f",i,author_first,author_last,shape,rate);
        }
    }


    /* Copy the random effects to the GPU */ 
    cudaMemcpy(rfx,Hrfx,HI*sizeof(double),cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    /* Copy the random effects in the  COVARIATES array */
    SaveRFX<<<CUBLAS_TMP,512>>>(rfx,COVARIATES);
    cudaDeviceSynchronize();


}



