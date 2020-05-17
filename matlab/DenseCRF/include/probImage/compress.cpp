#include "probimage.h"
#include <cstdio>

int main( int argc, char * argv[] ){
	if (argc<3){
		printf("Usage: %s file compressed_file\n", argv[0] );
	}
	ProbImage im;
	im.load( argv[1] );
	im.boostToProb();
	im.compress( argv[2], 0.001 );
	
	return 0;
}
