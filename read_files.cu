#include <stdlib.h>
#include <stdio.h>

extern int HI;      // the total number of point patterns
extern int HN;      // the total number of points
extern int HK_star; // the total number of covariates
extern int Hd;      // the dimensionality of the problem
extern int HV;      // the total number of elements in the grid

void read_files(int ** foci , float ** Z , int * foci_counts , int * brain , int* author) {
    int i , j; // those are for the for loops
    
    // read the file with the points
    FILE *file;
    file = fopen("./inputs/foci.txt","r");
    for (i=0 ; i<HN ; i++) {
        for (j=0 ; j<(Hd+1) ; j++) {
            if(!fscanf(file,"%d",&foci[i][j]))
                break;
        }
    }
    fclose(file);
    
    // read the design matrix
    file = fopen("./inputs/Z.txt","r");
    for (i=0 ; i<HI ; i++) {
        for (j=0 ; j<HK_star ; j++) {
            if(!fscanf(file,"%f",&Z[i][j]))
                break;
        }
    }
    fclose(file);

    // read the total number of points per pattern
    file = fopen("./inputs/counts.txt","r");
    for (i=0 ; i<HI ; i++) {
        if(!fscanf(file,"%d",&foci_counts[i]))
            break;
    }
    fclose(file);

    /* Binary vector indicating which voxels in the grid are within the brain mask */ 
    file = fopen("mask.txt","r");
    for (i=0 ; i<HV ; i++) {
        if(!fscanf(file,"%d",&brain[i]))
            break;
    }
    fclose(file);
    
    /* Read the publication identifier. Studies from the same paper must appear consecutive */
    file = fopen("./inputs/paper.txt","r");
    for (i=0 ; i<HI ; i++) {
        if(!fscanf(file,"%d",&author[i]))
            break;
    }
    fclose(file);

}