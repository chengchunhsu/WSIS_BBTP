classdef MSRC < Densecrf
	
	properties (SetAccess = protected)
		resize_factor;
		image_name;
		gt;
	end
	
	properties (Hidden)
		msrc_root;
	end

	methods (Static)
		% Wrapper for probImage decompress
		function decompressed =  decompress(file_path)
			
            
			addpath([fileparts(mfilename('fullpath')) '/include']);
			
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
		end

		% Returns all image names in a cell.
		function names = all_images(msrc_root)
			if nargin < 1
				msrc_root = ['..' filesep 'data' filesep 'MSRC'];
			end

			image_path = [msrc_root filesep 'Images'];
			f = dir([image_path filesep '*.bmp']);
			names = extractfield(f,'name');
			
			for n = 1:numel(names)
				[~,names{n}] = fileparts(names{n});
			end
		end	
	end
	
	methods
		function self = MSRC(image_name, resize_factor)
			
			if nargin < 2
				resize_factor = 1;
			end
			
			base_path = fileparts(mfilename('fullpath'));	
			unary_path = sprintf('%s/data/MSRC/unary/%s.c_unary',base_path,image_name);
			im_path = sprintf('%s/data/MSRC/Images/%s.bmp',base_path,image_name);
			
			unary = -single(MSRC.decompress(unary_path));
			im = uint8(imread(im_path));
			
			unary = imresize(unary,resize_factor);
			im = imresize(im,resize_factor);
			
			self = self@Densecrf(im, unary);
			self.resize_factor = resize_factor;
			self.image_name = image_name;
			
			% Load ground truth
			% MSRC
			msrc_root = [base_path  filesep 'data' filesep 'MSRC'];

			gt_folder = [msrc_root filesep 'GroundTruth'];
			
			% Prefer hq if it exist (not done for every image).
			gt_hq_folder = [msrc_root filesep 'SegmentationsGTHighQuality'];
			
			% Another set this one by Krähenbühl
			gt_hq_folder2 =[msrc_root filesep 'HighQuality'];
			
			if ~exist(gt_folder)
				disp 'WARNING: Missing GT folder';
			end
			
			if ~exist(gt_hq_folder)
				disp 'WARNING: Missing HQ GT folder (SegmentationsGTHighQuality)';
			end
			
			if ~exist(gt_hq_folder2)
				disp 'WARNING: Missing HQ GT folder (HighQuality)';
			end
			
			hq_gt_filename = [gt_hq_folder filesep self.image_name '_HQGT.bmp'];
			hq2_gt_filename = [gt_hq_folder2 filesep self.image_name '_GT.bmp'];
			gt_filename = [gt_folder filesep self.image_name '_GT.bmp'];
			
			if exist(hq_gt_filename)
				gt = imread(hq_gt_filename);
			elseif	exist(hq2_gt_filename)
				gt = imread(hq2_gt_filename);
			else
				if ~exist(gt_filename)
					error(sprintf('gt for %s is missing', self.image_name));
				end
				
				gt = imread(gt_filename);
			end
			
			% Calculate resize factor
			if (self.resize_factor ~= 1)
				gt = self.rescale_gt(gt, self.resize_factor);
			end
			
			self.gt = self.RGB2label(gt);
			self.msrc_root = msrc_root;
		end
		
		function score = score(self)
			score = 100*( sum(self.gt(:) == self.segmentation(:)) )/numel(self.segmentation);
		end
		
		function GT_resized = rescale_gt(~, GT, resize_factor)
				%as it sounds, re
				diff_c = []; %contain the different colors.
				sizes = size(GT);
				for i = 1:sizes(1)
					for j = 1:sizes(2)
						if ~mod(i,50)
							[i sizes(1)];
						end
						contains = 0;
						pixel_color = GT(i,j,:);
						pixel_color = pixel_color(:)';
						for k = 1:size(diff_c,1)
							
							if isequal(diff_c(k,:),pixel_color)
								contains = 1;
								break
							end
							
						end
						if ~contains
							diff_c = [diff_c; pixel_color];
						end
					end
				end
				diff_c;
				GT_resized = imresize(GT,resize_factor);
				
				
				sizes = size(GT_resized);
				for i = 1:sizes(1)
					for j = 1:sizes(2)
						best_distans = 100000000;
						pixel_color = GT_resized(i,j,:);
						pixel_color = pixel_color(:)';
						
						ind = 0;
						for k = 1:size(diff_c,1)
							if sum((double(diff_c(k,:))-double(pixel_color)).^2) < best_distans
								best_distans = sum((double(diff_c(k,:))-double(pixel_color)).^2);
								ind = k;
							end
							
						end
						ind;
						GT_resized(i,j,1) = diff_c(ind,1);
						GT_resized(i,j,2) = diff_c(ind,2);
						GT_resized(i,j,3) = diff_c(ind,3);
						
					end
				end
			end
			
			function L = RGB2label(~, IM)
				
				assert(ndims(IM) == 3);
				sz = size(IM);
				
				if ~(min(IM(:))>=0 &&  max(IM(:)) < 256)
					error('Wrong format,  IM \in [0,255]')
				end
				
				% Preallocate
				%labels to RBG fig 1 textonboost.
				%reading row before columns [1 2 3 4.....; 11 12...]
				
				lab_to_rgb = zeros(23,3);
				lab_to_rgb(1,:) = [128 0 0];
				lab_to_rgb(2,:) =  [0 128 0]; % Grass
				lab_to_rgb(3,:) =  [128 128 0];
				lab_to_rgb(4,:) =  [0 0 128]; % Cow
				lab_to_rgb(5,:) =  [0 128 128];
				lab_to_rgb(6,:) =  [128 128 128];
				lab_to_rgb(7,:) =  [192 0 0];
				lab_to_rgb(8,:) =  [64 128 0];
				lab_to_rgb(9,:) =  [192 128 0];
				lab_to_rgb(10,:) =  [64 0 128];
				lab_to_rgb(11,:) =  [192 0 128];
				lab_to_rgb(12,:) =  [64 128 128];
				lab_to_rgb(13,:) =  [192 128 128];
				lab_to_rgb(14,:) =  [0 64 0];
				lab_to_rgb(15,:) =  [128 64 0];
				lab_to_rgb(16,:) =  [0 192 0];
				lab_to_rgb(17,:) =  [128 64 128];
				lab_to_rgb(18,:) =  [0 192 128];
				lab_to_rgb(19,:) =  [128 192 128];
				lab_to_rgb(20,:) =  [64 64 0];
				lab_to_rgb(21,:) =  [192 64 0];
				lab_to_rgb(22,:) =	[0 0 0];
				lab_to_rgb(23,:) =  [64 0 0];
				
				% Reduce to 1D search
				findex = @(a) a(:,1) + a(:,2)*255 + a(:,3)*255^2;
				fast_index = findex(lab_to_rgb);
				fast_im = findex( double(reshape(IM, [],3)));
				
				% Put in place when duplicates are removed.
				%assert( numel(unique(fast_index)) == size(lab_to_rgb,1))
				L = zeros(size(fast_im,1),1);
				
				% Annotation is not consistent so need to add some duplicates
				tree = findex([128 128 0]);
				L(fast_im == tree) = 3;
				for i = 1:size(fast_index,1)
					L(fast_im == fast_index(i)) = i;
				end
								
				L = reshape(L, sz(1:2));
				
				% These labels lack unary cost
				L(L > 21) = 0;
			end

			function display(self)
				display@Densecrf(self);
				self.display_result_only();
			end

			function display_result_only(self)
				imagesc(self.segmentation);
				axis equal; axis off;
				caxis([1 21]);
			 title(sprintf('Energy:  %2.2e\nLower bound: %2.2e\n Gap: %2.2e\nScore: %g\nSolver: %s', ...
					self.energy, self.lower_bound, self.energy_gap, self.score(), self.solver), ...
					'Units', 'normalized', 'Position', [1 1], 'HorizontalAlignment', 'right');
			end
		end
	end