double2 P(double2 in){
	return (double2) (-in.x-in.y,in.x*in.y);
}

kernel void schlieren( global uchar* schlieren, const double Scale, const int Resolution, const int Iterations, const double vx, const double vy)
{
	const int idx = get_global_id(0);

	const int i = idx % Resolution;
	const int j = idx / Resolution;
			
	double delta = 0.5 * (Scale / Resolution);

	double2 pos = (double2) (((double)i / Resolution - 0.5) * Scale + delta - vx, (0.5 - (double)j / Resolution) * Scale - delta - vy);
	
	double2 posdx = pos + (double2) (delta, 0);
	double2 posdy = pos + (double2) (0, delta);	
	double2 pos_dx = pos + (double2) (-delta, 0);
	double2 pos_dy = pos + (double2) (0, -delta);	
	
	for (int k = 0; k < Iterations; k++) {

		if (sign(pos_dx.x) != sign(posdx.x) || sign(pos_dy.x) != sign(posdy.x)) { //VzW
			schlieren[idx] = 1;
			return;
		}

		posdx = P(posdx);
		posdy = P(posdy);
		pos_dx = P(pos_dx);
		pos_dy = P(pos_dy);
	
	}


	schlieren[idx] = 0;
}


kernel void scaledown(global uchar * oldbuffer, global uchar * newbuffer, const int oldres){ 

	const int idx = get_global_id(0);
	int newres = oldres / 2;
	
	const int i = idx % newres;
	const int j = idx / newres;

	const int oldi = i * 2;
	const int oldj = j * 2;

	newbuffer[idx] = ((oldbuffer[oldres*oldj + oldi] == 1) | (oldbuffer[oldres*(oldj+1) + oldi] == 1) | (oldbuffer[oldres*(oldj) + oldi + 1] == 1) | (oldbuffer[oldres*(oldj+1) + oldi + 1] == 1))? 1 : 0;

}

kernel void sum(const int res, global int *oldbuffer, global int *newbuffer){ 
	const int idx = get_global_id(0);
	int newres = res / 2;
	
	const int i = (idx % newres);
	const int j = idx / newres;
	const int base_ = i*2 + j*2*res;

	newbuffer[idx] = oldbuffer[base_] + oldbuffer[base_+1] + oldbuffer[base_+res] + oldbuffer[base_+res+1];
}
kernel void firstsum(const int res, global uchar *oldbuffer, global int *newbuffer){ 
	const int idx = get_global_id(0);
	int newres = res / 2;
	
	const int i = (idx % newres);
	const int j = idx / newres;
	const int base_ = i*2 + j*2*res;

	newbuffer[idx] = oldbuffer[base_] + oldbuffer[base_+1] + oldbuffer[base_+res] + oldbuffer[base_+res+1];
}
