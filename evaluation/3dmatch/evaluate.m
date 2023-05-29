% Script to evaluate .log files for the geometric registration benchmarks,
% in the same spirit as Choi et al 2015. Please see:
%
% http://redwood-data.org/indoor/regbasic.html
% https://github.com/qianyizh/ElasticReconstruction/tree/master/Matlab_Toolbox


descriptorName = 'reg'; % 3dmatch, spin, fpfh

% Locations of evaluation files
dataPath = '../../saved/log_hf_0922_114143_H3DNet_v8_X3DMatchFragmentDataLoader_max_k_0.2_4_corner_sz_4_cluster_radius_0.05_fix_int_v2/log_result_250';
         
% Real data benchmark
sceneList = {
              '7-scenes-redkitchen-evaluation', ...
              'sun3d-home_at-home_at_scan1_2013_jan_1-evaluation', ...
              'sun3d-home_md-home_md_scan9_2012_sep_30-evaluation', ...
              'sun3d-hotel_uc-scan3-evaluation', ...
              'sun3d-hotel_umd-maryland_hotel1-evaluation', ...
              'sun3d-hotel_umd-maryland_hotel3-evaluation', ...
              'sun3d-mit_76_studyroom-76-1studyroom2-evaluation', ...
              'sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika-evaluation'
             };
         
% Load Elastic Reconstruction toolbox
addpath(genpath('external'));

% Compute precision and recall
totalRecall = []; totalPrecision = [];
totalGt = 0;
totalTP = 0;
for sceneIdx = 1:length(sceneList)
    scenePath = fullfile(dataPath,sceneList{sceneIdx});
    % gtPath = fullfile('../gt_result',sceneList{sceneIdx});
    gtPath = fullfile('../../data/3dmatch/geometric_registration_adaptive',sceneList{sceneIdx});
    
    % Compute registration error
    gt = mrLoadLog(fullfile(gtPath,'gt.log'));
    gt_info = mrLoadInfo(fullfile(gtPath,'gt.info'));
    result = mrLoadLog(fullfile(scenePath,sprintf('%s.log',descriptorName)));
    [recall,precision,gt_num] = mrEvaluateRegistration(result,gt,gt_info);
    totalRecall = [totalRecall;recall];
    totalPrecision = [totalPrecision;precision];
    totalGt = totalGt + gt_num;
    totalTP = totalTP + round(gt_num * recall);
end
totalRecall
fprintf('Mean registration recall: %f precision: %f\n',mean(totalRecall),mean(totalPrecision));
fprintf('True average recall: %f (%d/%d)\n',totalTP/totalGt,totalTP, totalGt);
