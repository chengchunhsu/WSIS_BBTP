% Wrapper for probImage decompress
function decompressed =  decompress(file_path)

addpath('auxilary');

if ~(exist(file_path))
  error('Unary file given does not exist')
end

cpp_file = 'decompress_mex.cpp';
out_file = 'decompress_mex';

extra_arguments = {};

% Additional files to be compiled.
dec_dir = ['probImage' filesep];
sources = {[dec_dir 'probimage.cpp']};

% Only compile if files have changed
compile(cpp_file, out_file, sources, extra_arguments)

decompressed =  decompress_mex(file_path);
