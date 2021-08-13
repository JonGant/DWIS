#!/bin/bash
dir="nifti_images/"
for f in ${dir}jacks.nii.gz
do
	f=${f%.nii.gz}
	echo "Processing $f file.."
	mrconvert ${f}.nii.gz -fslgrad ${f}.bvec ${f}.bval ${f}.mif -force
	# dwi2response tournier ${f}.mif ${f}_wm_rf.txt -force
	dwi2fod msmt_csd ${f}.mif ${dir}l29_r3_roi15_wm_rf.txt ${f}_wm_fod.mif ${dir}l29_r3_roi15_gm_rf.txt ${f}_gm_fod.mif ${dir}l29_r3_roi15_csf_rf.txt ${f}_csf_fod.mif -force
	mrconvert ${f}.mif -coord 3 0 -axes 0,1,2 ${f}_wb_mask_temp.mif -force
	mrcalc ${f}_wb_mask_temp.mif -round ${f}_wb_mask.mif -force
	rm ${f}_wb_mask_temp.mif
	tckgen -algorithm sd_stream ${f}_wm_fod.mif -select 0 -seed_grid_per_voxel ${f}_wb_mask.mif 5 ${f}_wb.tck -force
	tck2connectome ${f}_wb.tck ${f}_roi.nii.gz ${f}_wb_sn.txt -symmetric -zero_diagonal -out_assignments ${f}_assignments.txt -force
	mkdir ${f}_exemplars/
	connectome2tck ${f}_wb.tck ${f}_assignments.txt ${f}_exemplars/edge -force
	rm ${f}_wb.tck
done
