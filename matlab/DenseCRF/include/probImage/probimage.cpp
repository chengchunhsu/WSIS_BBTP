#include "probimage.h"
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
// Endian convertion

#ifdef _WIN32
	#include "winsock2.h"
	#include <stdint.h>
	#pragma comment(lib, "ws2_32.lib")
#else
	#include <netinet/in.h>
#endif

ProbImage::ProbImage() :data_(NULL),width_(0),height_(0),depth_(0){
}
ProbImage::ProbImage(const ProbImage& o) : width_( o.width_ ), height_( o.height_ ), depth_( o.depth_ ){
	data_ = new float[ width_*height_*depth_ ];
	memcpy( data_, o.data_, width_*height_*depth_*sizeof(float) );
}
/*
ProbImage& ProbImage::operator=(const ProbImage& o) {
	width_ = o.width_; height_ = o.height_; depth_ = o.depth_;
	if( data_ ) delete[] data_;
	data_ = new float[ width_*height_*depth_ ];
	memcpy( data_, o.data_, width_*height_*depth_*sizeof(float) );
}
*/
ProbImage::~ProbImage(){
	if( data_ ) delete[] data_;
}
static void readBuf32( FILE * fp, unsigned int size, uint32_t * buf ){
	fread( buf, sizeof(*buf), size, fp );
	for( int i=0; i<size; i++ )
		buf[i] = ntohl( buf[i] );
}

static void writeBuf32( FILE * fp, unsigned int size, uint32_t * buf ){
	uint32_t sbuf[(1<<13)];
	for( int i=0; i<size; i+=(1<<13) ){
		for( int j=0; j<(1<<13) && i+j<size; j++ )
			sbuf[j] = htonl( buf[i+j] );
		fwrite( sbuf, sizeof(*sbuf), (size-i) < (1<<13) ? (size-i) : (1<<13), fp );
	}
}
void ProbImage::load(const char* file) {
	FILE* fp = fopen( file, "rb" );
	uint32_t buf[4];
	readBuf32( fp, 3, buf );
	width_ = buf[0]; height_ = buf[1]; depth_ = buf[2];
	if( data_ ) delete[] data_;
	data_ = new float[ width_*height_*depth_ ];
	readBuf32( fp, width_*height_*depth_, (uint32_t*)data_ );
	fclose( fp );
}

void ProbImage::save(const char* file) {
	FILE* fp = fopen( file, "wb" );
	uint32_t buf[] = {width_, height_, depth_};
	writeBuf32( fp, 3, buf );
	writeBuf32( fp, width_*height_*depth_, (uint32_t*)data_ );
	fclose( fp );
}

static int cmpKey( const void * a, const void * b ){
	const int * ia = (const int*)a, *ib = (const int*)b;
	for( int i=0;; i++ )
		if (ia[i] < ib[i])
			return -1;
		else if (ia[i] > ib[i])
			return 1;
	return 0;
}
void ProbImage::boostToProb() {
	for( int i=0; i<width_*height_; i++ ){
		float * dp = data_ + i*depth_;
		float mx = dp[0];
		for( int j=1; j<depth_; j++ )
			if (mx < dp[j])
				mx = dp[j];
		float nm = 0;
		for( int j=0; j<depth_; j++ )
			nm += (dp[j] = exp( (dp[j]-mx) ) );
		nm = 1.0 / nm;
		for( int j=0; j<depth_; j++ )
			dp[j] *= nm;
	}
}
void ProbImage::probToBoost() {
	for( int i=0; i<width_*height_; i++ ){
		float * dp = data_ + i*depth_;
		for( int j=0; j<depth_; j++ )
			dp[j] = log( dp[j] );
	}
}

void ProbImage::compress(const char* file, float eps) {
	// Compress using a lattice (A* should be fine)
	// For now just use the Z lattice, because I'm lazy
	float inv_esp = 1.0 / eps;
	
	int KS = depth_;
	int * keys = new int[ width_*height_*(KS+1) ];
	
	for( int i=0; i<width_*height_; i++ ){
		// Elevate the point
		for( int j=0; j<depth_; j++ )
			keys[i*(KS+1)+j] = (int) (inv_esp * data_[i*depth_+j]+0.5);
		keys[i*(KS+1)+KS] = i;
	}
	qsort( keys, width_*height_, (KS+1)*sizeof(int), cmpKey );
	
	int M=1;
	for( int i=1; i<width_*height_; i++ ){
		bool is = 1;
		for( int j=0; is && j<KS; j++ )
			is = (keys[i*(KS+1) + j] == keys[(i-1)*(KS+1) + j]);
		M+=!is;
	}
	uint32_t * ids = new uint32_t[width_*height_];
	int32_t * ukeys = new int32_t[M*depth_];
	for( int i=0, k=0; i<width_*height_; i++ ){
		bool is = (i>0);
		for( int j=0; is && j<KS; j++ )
			is = (keys[i*(KS+1) + j] == keys[(i-1)*(KS+1) + j]);
		if(!is){
			// Find the lattice coordinate
			for( int j=0; j<KS; j++ )
				ukeys[k*depth_+j] = keys[i*(KS+1)+j];
			k++;
		}
		ids[ keys[i*(KS+1)+KS] ] = k-1;
	}
	
	delete[] keys;
	
	
	
	FILE* fp = fopen( file, "wb" );
	uint32_t buf[] = {width_, height_, depth_, M};
	writeBuf32( fp, 4, buf );
	writeBuf32( fp, 1, (uint32_t*)&eps );
	writeBuf32( fp, M*depth_, (uint32_t*)ukeys );
	writeBuf32( fp, width_*height_, ids );
	
	fclose( fp );
	delete[] ukeys;
	delete[] ids;
}

void ProbImage::decompress(const char* file) {
	FILE* fp = fopen( file, "rb" );
	uint32_t buf[5];
	readBuf32( fp, 5, buf );
	width_ = buf[0]; height_ = buf[1]; depth_ = buf[2];
	int M = buf[3];
	float eps = *(float*)(buf+4);
	
	uint32_t * ids = new uint32_t[width_*height_];
	int32_t * ukeys = new int32_t[M*depth_];
	readBuf32( fp, M*depth_, (uint32_t*)ukeys );
	readBuf32( fp, width_*height_, ids );
	
	if( data_ ) delete[] data_;
	data_ = new float[ width_*height_*depth_ ];
	
	for( int i=0; i<width_*height_; i++ ){
		int32_t * k = ukeys + ids[i]*depth_;
		for( int j=0; j<depth_; j++ )
			data_[ i*depth_ + j ] = k[j]*eps;
	}
	
	fclose( fp );
	delete [] ids;
	delete [] ukeys;
}
