#include "probimage.h"
#include <cstdio>

int main( int argc, char * argv[] ){
	if (argc<3){
		printf("Usage: %s compressed_file file\n", argv[0] );
	}
	ProbImage im;
	im.decompress( argv[1] );
	//  im.probToEnergy();
	im.save( argv[2] );
	
	return 0;
}
