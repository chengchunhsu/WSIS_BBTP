% Solves densecrf problem described in:
%
% Philipp Krähenbühl and Vladlen Koltun
% Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials
% NIPS 2011 
%
%	Solvers:
%	mean_field          : Krähenbühls' mean field approximation implementation (uses fast filtering)
%	mean_field_explicit : Slower but more exact mean field approximation implementation (perform all summations)
%	threshold           : thresholds the unary cost
%	trws                : Convergent Tree-reweighted Message Passing .  
% 
%	Debug method:
%	random_solution     : Set a random segmentation
%	most_probable_label : Set the segmentation to the single label with lowest energy.
%
% Remarks:
% The energy reported is calculated via approximate filtering.
% Exact energy may be calculate via the exact_energy method
%
% WARNING: 
% If NormalizationType is changed the problem mean_field solves is 
% redefined and the other solvers solves a different problem.
%
classdef Densecrf < handle
	% Settings
	properties		
		gaussian_x_stddev = 3;
		gaussian_y_stddev = 3;
		gaussian_weight = 1; 

		bilateral_x_stddev = 60;
		bilateral_y_stddev = 60;
		bilateral_r_stddev = 10;
		bilateral_g_stddev = 10;
		bilateral_b_stddev = 10;
		bilateral_weight = 1; 
		
		debug = false;
		iterations = 100;
		
		% Used for trws()
		% pairwise cost which are lower then this are not added to the cost function.
		% For larger images this can be used to limit the memory usage. 
		% The energy will not be correct but the lower bound will still be valid.
		min_pairwise_cost = 0;
	
		segmentation = [];

		% Used for mean_field()
		%	NO_NORMALIZATION,    // No normalization whatsoever (will lead to a substantial approximation error)
		% NORMALIZE_BEFORE,    // Normalize before filtering (Not used, just there for completeness)
	  % NORMALIZE_AFTER,     // Normalize after filtering (original normalization in NIPS 11 work)
		% NORMALIZE_SYMMETRIC, // Normalize before and after (ICML 2013, low approximation error and preserves the symmetry of CRF)
		NormalizationType = 'NO_NORMALIZATION';

		% Virtual
		im;
		unary;

		% Keep track on which algorithm gave the stored result.
		solver = '';
	end
	
	properties (SetAccess = protected)
		im_stacked;
		unary_stacked;
		lower_bound = -inf;
		energy = nan;
		energy_gap;
		optimization_time = -1;
	end
	
	properties (Hidden)
		image_size;
		get_energy = true;
	end
	
	methods (Static)
		% Restack 3D matrix s.t
		% x0y0z0 x0y0z1 , .... x1y0z0,x1y0z1
		function out = color_stack(in)	
			assert(ndims(in) == 3);
			out = zeros(numel(in),1);
			
			colors = size(in,3);
			
			for c = 1:colors
				out(c:colors:end) = reshape(in(:,:,c),[],1);
			end
		end
	
		% Inverse of color_stack
		function out = inverse_color_stack(in, image_size)
			assert(isvector(in));
			colors = image_size(3);
			
			assert(mod(numel(in),colors) == 0);
			assert(numel(image_size) == 3);
			
			out = zeros(image_size);
			for c = 1:colors
				out(:,:,c) = reshape(in(c:colors:end),image_size(1:2));
			end
		end
		
		
	end
		
	methods

		% Gather and format
		function settings = gather_settings(self)

			settings.gaussian_x_stddev = self.gaussian_x_stddev;
			settings.gaussian_y_stddev = self.gaussian_y_stddev;
			settings.gaussian_weight =  self.gaussian_weight;
			
			settings.bilateral_x_stddev = self.bilateral_x_stddev;
			settings.bilateral_y_stddev = self.bilateral_y_stddev;
			settings.bilateral_r_stddev = self.bilateral_r_stddev;
			settings.bilateral_g_stddev = self.bilateral_g_stddev;
			settings.bilateral_b_stddev = self.bilateral_b_stddev;
			settings.bilateral_weight = self.bilateral_weight;
			settings.min_pairwise_cost = self.min_pairwise_cost;
			settings.NormalizationType = self.NormalizationType;

			settings.debug = logical(self.debug);
			settings.iterations = int32(self.iterations);
        end
		
        function self =  Densecrf()
        end
        
        function self = SetOpts(self, Opts)
            OptsFieldNames = fieldnames(Opts);
            for i = 1:length(OptsFieldNames)
                assert(isprop(self, OptsFieldNames{i}), ['Error Property: ' OptsFieldNames{i}])
                self.(OptsFieldNames{i}) = Opts.(OptsFieldNames{i});
            end
        end
        
        function self =  SetImage(self, im)
            if ~isa(im,'uint8')
                warning('Image is not unsgined 8 bit int, converting.');
            end
            self.image_size = uint32(size(im));
            assert(numel(self.image_size) == 3);
            self.im = im;
            self.get_energy = false;
            self.segmentation = ones(self.image_size(1:2));
            self.get_energy = true;
        end
        
        function self = SetUnary(self, unary)
            if ~isa(unary,'single')
                warning( 'Unary cost must be float/single, converting.');
            end
            self.unary = unary;
        end
        

		% Compile if need be
		function compile(~, file_name)
			my_name = mfilename('fullpath');
			my_path = [fileparts(my_name) filesep];
			eigen_path = [my_path 'include' filesep 'densecrf' filesep 'include' filesep];
			lbfgs_include_path = [my_path 'include' filesep 'densecrf' filesep 'external' filesep 'liblbfgs' filesep  'include' filesep];

			cpp_file = [file_name '_mex.cpp'];
			out_file = [file_name '_mex'];
			
			extra_arguments = {};
			extra_arguments{end+1} = ['-I' my_path];
			extra_arguments{end+1} = ['-I' eigen_path];
			extra_arguments{end+1} = ['-I' lbfgs_include_path];
			
			if ~ispc
				extra_arguments{end+1} = ['-lgomp'];
			end
			
			% Additional files to be compiled.
			mf_dir = ['densecrf' filesep 'src' filesep];
			trws_dir =  ['TRW_S-v1.3' filesep];
			lbfgs_dir = ['densecrf' filesep 'external'  filesep 'liblbfgs' filesep 'lib' filesep];
			maxflow_dir = ['maxflow-v3.03.src' filesep];

			sources = {[mf_dir 'util.cpp'], ...
				[mf_dir 'densecrf.cpp'], ...
				[mf_dir 'labelcompatibility.cpp'], ...
				[mf_dir 'objective.cpp'], ...
				[mf_dir 'optimization.cpp'], ...
				[mf_dir 'pairwise.cpp'], ...
				[mf_dir 'permutohedral.cpp'], ...
				[mf_dir 'unary.cpp'], ...
				[lbfgs_dir 'lbfgs.cpp'], ...
				[trws_dir 'minimize.cpp'], ...
				[trws_dir 'MRFEnergy.cpp' ], ...
				[trws_dir 'ordering.cpp'], ...
				[trws_dir 'treeProbabilities.cpp' ], ...
				[maxflow_dir 'maxflow.cpp'], ...
				[maxflow_dir 'graph.cpp']};

			% Only compile if files have changed
			compile(cpp_file, out_file, sources, extra_arguments)
		end
		
		function segmentation = mean_field_explicit(self)
			settings = self.gather_settings;
			settings.solver = 'mean_field_explicit';
			self.compile('densecrf');
			
			t = tic
			segmentation =  densecrf_mex(self.im_stacked, self.unary_stacked, self.image_size, settings);
			self.optimization_time = toc(t);

			segmentation = segmentation+1;
			
			self.segmentation = segmentation;
			self.solver = 'mean field approximation (explicit summations)';
		end


		function segmentation = threshold(self)
			t = tic;
			[~,segmentation] = min(self.unary,[],3);
			self.optimization_time = toc(t);
			self.segmentation = segmentation;
			self.solver = 'threshold';
		end
		
		function segmentation = mean_field(self)
			settings = self.gather_settings;
			settings.solver = 'mean_field';
			self.compile('densecrf');
			
			t = tic;
			[segmentation, energy, bound] =  densecrf_mex(self.im_stacked, self.unary_stacked, self.image_size, settings);
			self.optimization_time = toc(t);

			segmentation = segmentation+1;

			tmp = self.get_energy;
			self.get_energy = false;
			self.segmentation = segmentation;
			self.get_energy = tmp;
			self.energy = energy;
			self.lower_bound = bound;
		
			self.solver = 'mean field';
		end

		function [segmentation, energy, lower_bound] = trws(self)
			if (self.num_labels() < 3) 
				error('For binary problems use graph_cuts!')
			end

			settings = self.gather_settings;
			settings.solver = 'trws';
			self.compile('densecrf');
			
			t = tic;
			[segmentation, energy, lower_bound] =  densecrf_mex(self.im_stacked, self.unary_stacked, self.image_size, settings);
			self.optimization_time = toc(t);

			segmentation = segmentation+1;
			self.segmentation = segmentation;
			self.lower_bound = lower_bound;
			self.solver = 'TRW-S';
		end

		function segmentation = graph_cuts(self) 
			if (self.num_labels() > 2) 
				error('Graph cut only works for 2-label problems');
			end

			settings = self.gather_settings;
			settings.solver = 'graph_cuts';
			self.compile('densecrf');

			% Only postive weights
			offset = - min(self.unary_stacked(:));
			
			t = tic;
			segmentation =  densecrf_mex(self.im_stacked, self.unary_stacked + offset, self.image_size, settings);
			self.optimization_time = toc(t);

			segmentation = segmentation+1;
			self.segmentation = segmentation;
			self.lower_bound = self.energy; % Always global optima.
			self.solver = 'Graph cuts';
		end

		% Calculate exact energy of current solution
		function energy = calculate_energy(self)
			self.compile('energy');
			settings = self.gather_settings;
			segmentation = int16(self.segmentation - 1);

			[~, energy] =  energy_mex(self.im_stacked, self.unary_stacked, self.image_size, segmentation, settings);
			self.energy = energy;
		end

		% Calculate exact energy by summing of all pairs (this is very slow)
		function [exact_energy, mf_energy] = calculate_exact_energy(self)
			self.compile('energy');
			settings = self.gather_settings;
			settings.calculate_exact_energy = true;

			segmentation = int16(self.segmentation - 1);
			[exact_energy, mf_energy] =  energy_mex(self.im_stacked, self.unary_stacked, self.image_size, segmentation, settings);
			self.energy = mf_energy;
		end

		
		function display(self)		
			subplot(1,2,1)
			imshow(double(self.im)/256)
			title('Image');

			if (~isempty(self.segmentation))
				subplot(1,2,2);
				imagesc(self.segmentation);
				axis equal; axis off;
			 title(sprintf('Energy:  %2.2e\nLower bound: %2.2e\n Gap: %2.2e\nSolver: %s', ...
					self.energy, self.lower_bound, self.energy_gap,  self.solver), ...
					'Units', 'normalized', 'Position', [1 1], 'HorizontalAlignment', 'right');
			end
		
			details(self);
		end

		function num_labels = num_labels(self)
			num_labels = size(self.unary,3);
		end

		% Generate a random solution.
		function segmentation = random_solution(self, seed)
			if nargin == 2
				rng(seed)
			end
				
			segmentation = ceil(rand(self.image_size(1:2))*self.num_labels());
			self.segmentation = segmentation;
			self.solver = 'random solution';
		end
		
		function segmentation = most_probable_label(self)
			segmentation = ones(self.image_size(1),self.image_size(2));

			[~,threshold] = min(self.unary,[],3);
			segmentation(:) = mode(threshold(:));

			self.segmentation = segmentation;
			self.solver = 'most probable label';
		end

		% No regularization cost
		function lower_bound = unary_lower_bound(self)
			lower_bound = sum(sum(min(self.unary,[],3)));
		end

		% set/get methods
		function gap = get.energy_gap(self)
			gap = (self.energy - self.lower_bound);
		end

		function set.im(self, im)
			self.im = im;
			
			% Stacking s.t. colors is contiguous in memory
			self.im_stacked = uint8(Densecrf.color_stack(im));
		end

		function set.unary(self, unary)
			self.unary = unary;
			self.unary_stacked = single(Densecrf.color_stack(unary));
		end
		
		function set.segmentation(self, segmentation)

			if ~all( size(segmentation) ==  self.image_size(1:2))
				error('Segmentation must be of same size as image.');
			end

			if min(segmentation(:) < 1)
				error('Segmentation entries should be 1,...,num labels.');
			end

			if max(segmentation(:) > size(self.unary,3))
				error('Segmentation entries should be 1,...,num labels.');
			end		

			if (norm(round(segmentation(:)) - segmentation(:)) > 0)
				error('Segmentation entries must be integers.');
			end

			self.segmentation = segmentation;
			
			if (self.get_energy)
				self.calculate_energy();
			else
				self.energy = nan;
			end

			self.lower_bound = self.unary_lower_bound();
			self.solver = '';
		end
		
		function set.NormalizationType(self, NormalizationType)

			ok_values = {'NO_NORMALIZATION','NORMALIZE_BEFORE','NORMALIZE_AFTER','NORMALIZE_SYMMETRIC'};
			hit = false;

			for v = 1:numel(ok_values)
				if strcmp(ok_values{v},NormalizationType)
					hit = true;
					break;
				end
			end
			
			if (~hit)
				error('Allowed values: NormalizationType={%s, %s, %s, %s} ', ok_values{:})
			end

			self.NormalizationType = NormalizationType;
		end
	end	
end