double2 P(double2 in){
	return (double2) (-in.x-in.y,in.x*in.y);
}

kernel void schlieren( global uchar* schlieren, const double Scale, const int Resolution, const int Iterations, const double vx, const double vy)
{
	const int idx = get_global_id(0);

	const int i = idx % Resolution;
	const int j = idx / Resolution;

	double2 pos = (double2) (((double)i / Resolution - 0.5) * Scale - vx, (0.5 - (double)j / Resolution) * Scale - vy);
	
	double delta = 1 * (Scale / Resolution);
	
	double2 posdx = pos + (double2) (delta, 0);
	double2 posdy = pos + (double2) (0, delta);	
	
	for (int k = 0; k < Iterations; k++) {

		if (sign(pos.x) != sign(posdx.x) || sign(pos.x) != sign(posdy.x)) { //VzW
			schlieren[idx] = 1;
			return;
		}
		pos = P(pos);
		posdx = P(posdx);
		posdy = P(posdy);
	
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






/*
kernel void twotothenegativek( global bool* schlieren, const double Scale, const int Resolution, const int Iterations, const double vx, const double vy) {

	const int idx = get_global_id(0);

	const int i = idx % Resolution;
	const int j = idx / Resolution;

	double x = ((double)i / Resolution - 0.5) * Scale - vx;
	double y = (0.5 - (double)j / Resolution) * Scale - vy;
	
	double delta = 0.5  * (Scale / Resolution);
		
	for (int k = 0; k < Iterations; k++) {
		
		if(sign((x - delta)-(1.0 - pow(2.0, -(double)k))) != sign(((x + delta)-(1.0 - pow(2.0, -(double)k))))){
			schlieren[idx] = true;	
			return;
		}
	
	}

	schlieren[idx] = false;

}
*/