#pragma once

class ProbImage{
protected:
	float * data_;
	int width_, height_, depth_;
public:
	ProbImage();
	ProbImage( const ProbImage & o );
	~ProbImage();
	//ProbImage & operator=( const ProbImage & o );
	
	// Load and save
	void load( const char * file );
	void save( const char * file );
	void compress( const char * file, float eps );
	void decompress( const char * file );
	
	// Properties
	int width() const { return width_; }
	int height() const { return height_; }
	int depth() const { return depth_; }
	
	// Conversion operations
	void boostToProb();
	void probToBoost();
	
	// Data access
	const float & operator()( int i, int j, int k ) const { return data_[(j*width_+i)*depth_+k]; }
	float & operator()( int i, int j, int k ) { return data_[(j*width_+i)*depth_+k]; }
	const float * data() const { return data_; }
	float * data() { return data_; }
};
