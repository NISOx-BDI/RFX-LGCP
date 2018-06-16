#include <stdlib.h>
#include <math.h>

void distance(float *dist,int dim,int *m,int *n) {
	int i,j,k,h=0;
	float x[3],dm[3];
	float voxel_side = 2.0f;
		

	h = 0;
	dm[0] = (float)m[0];
	dm[1] = (float)m[1];
	dm[2] = (float)m[2];
			
	for (i=0;i<m[0];i++) {
		x[0] = (float)i;
		if (dm[0]/2. < x[0] && x[0] < dm[0])
			x[0] -= dm[0];
		else if (dm[0]/2 < -x[0] && -x[0] < dm[0])
			x[0] += dm[0];
		x[0] /= (float)n[0];
				
		for (j=0;j<m[1];j++) {
			x[1] = (float)j;
			if (dm[1]/2. < x[1] && x[1] < dm[1])
				x[1] -= m[1];
			else if (dm[1]/2. < -x[1] && -x[1] < dm[1])
				x[1] += dm[1];
			x[1] /= (float)n[1];
					
			for (k=0;k<m[2];k++) {
				x[2] = (float)k;
				if (dm[2]/2. < x[2] && x[2] < dm[2])
					x[2] -= dm[2];
				else if (dm[2]/2. < -x[2] && -x[2] < dm[2])
					x[2] += dm[2];
				x[2] /= (float)n[2];
				dist[h] = sqrt( pow(voxel_side*n[0]*x[0],2)+pow(voxel_side*n[1]*x[1],2)+pow(voxel_side*n[2]*x[2],2)   );
                h++;
			}
		}
	}
}