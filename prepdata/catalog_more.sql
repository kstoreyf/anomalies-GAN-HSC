SELECT

    -- Basic information
    f1.object_id, f1.ra, f1.dec, f1.tract, f1.patch, f1.parent_id,
    
    m.g_blendedness_abs_flux, m.g_blendedness_flag,
    m.r_blendedness_abs_flux, m.r_blendedness_flag,
    m.i_blendedness_abs_flux, m.i_blendedness_flag, 
    m.z_blendedness_abs_flux, m.z_blendedness_flag, 
    m.y_blendedness_abs_flux, m.y_blendedness_flag, 
    
        -- Shape of the CModel model
    m.i_cmodel_exp_ellipse_11, m.i_cmodel_exp_ellipse_22, m.i_cmodel_exp_ellipse_12,
    m.i_cmodel_dev_ellipse_11, m.i_cmodel_dev_ellipse_22, m.i_cmodel_dev_ellipse_12,
    m.i_cmodel_ellipse_11, m.i_cmodel_ellipse_22, m.i_cmodel_ellipse_12,
    m.r_cmodel_exp_ellipse_11, m.r_cmodel_exp_ellipse_22, m.r_cmodel_exp_ellipse_12,
    m.r_cmodel_dev_ellipse_11, m.r_cmodel_dev_ellipse_22, m.r_cmodel_dev_ellipse_12,
    m.r_cmodel_ellipse_11, m.r_cmodel_ellipse_22, m.r_cmodel_ellipse_12

        
FROM
    pdr2_wide.forced AS f1
    LEFT JOIN pdr2_wide.meas AS m USING (object_id)

WHERE
    
    -- Make sure we only select the primary target
    f1.isprimary = True
AND f1.nchild = 0

    -- HSC Wide is separated into 7 sub-regions from W01 to W07
    -- You can only select objects from one region using :
-- AND s18a_wide.search_w02(object_id)

    -- Rough FDFC cuts
AND f1.g_inputcount_value >= 3
AND f1.r_inputcount_value >= 3
AND f1.i_inputcount_value >= 3
AND f1.z_inputcount_value >= 3
AND f1.y_inputcount_value >= 3

    -- If you want to select star or galaxy
    -- Extended objects = 1; Point source = 0
-- AND f1.i_extendedness_value = 1
-- AND f1.r_extendedness_value = 1

AND NOT f1.g_pixelflags_bright_objectcenter
AND NOT f1.r_pixelflags_bright_objectcenter
AND NOT f1.i_pixelflags_bright_objectcenter
AND NOT f1.z_pixelflags_bright_objectcenter
AND NOT f1.y_pixelflags_bright_objectcenter

AND NOT f1.g_pixelflags_bright_object
AND NOT f1.r_pixelflags_bright_object
AND NOT f1.i_pixelflags_bright_object
AND NOT f1.z_pixelflags_bright_object
AND NOT f1.y_pixelflags_bright_object

    -- CModel magnitude limited
AND f1.i_cmodel_mag < 20.5 
AND f1.i_cmodel_mag >= 20.0
