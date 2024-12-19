




;*************************************************


;************************************************
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl" 
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
;load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"   
;load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"
;************************************************
;----------------------------------------------------------------------
; from panel_29.ncl plus https://www.ncl.ucar.edu/Support/talk_archives/2014/0250.html
; also modified using https://www.ncl.ucar.edu/Applications/Scripts/panel_23.ncl
;----------------------------------------------------------------------
;     Local function that extracts information from the
;     contour object attribute associated with the input plot. 
;     This makes the code a bit cleaner also.  
;----------------------------------------------------------------------
undef("createLabelBar")
function createLabelBar(wks: graphic, plt:graphic, lbres:logical)
begin
   getvalues plt               ; object from which info extracted
      "cnFillColors"   : fill_colors   ; get the colors used
      "cnLineLabelStrings" : lbstrings     ; get the labels used
   end getvalues
   
   delete(lbres@lbFillColors)
   lbres@lbFillColors = fill_colors    ; associate with input lnres argument
   lbid = gsn_create_labelbar(wks,dimsizes(lbstrings)+1,lbstrings,lbres)
   
   ; if you reuse this function in the same script but for contours with a different number of levels, you will get errors unless you delete the new vars
   delete(fill_colors)
   delete(lbstrings)
   
   return(lbid)
end



begin
 

;;;;;"net" is MASS or NUM, from the command line

   Nx =  37
   pres = 1

   levplot=  (/71, 55, 43/)
   tit2 = "diff_commonlbar"
   mod_name = "MAMnet_test"
   dlev =  1
   if (pres .gt. 0) then
     delete(levplot)
     delete(dlev)
     levplot=  (/950.0, 500.0, 250.0/)
     tit2 = "diff_pres_commonlbar"
     mod_name =  "MAMnet_test_pres"
     dlev = 25.0 
   
   end if 
   
   
   mod_tit =  "sigma_W"
   maxlogN =  0.1
   minlogN  = 0.0 
 
  lv_off =0
  wks_type = "png"  
 
  
    colormp = "wh-bl-gr-ye-re" 
 ; colormp =  "BlAqGrYeOrReVi200"
 ; colormp =  "BlAqGrYeOrRe"
 ; colormp ="WhiteBlueGreenYellowRed"
 ;colormp =  "BlueWhiteOrangeRed"
 ;colormp = "MPL_coolwarm"
; colormp =  "precip3_16lev"
 
  wks_type@wkPaperSize = "A4"
 ;gsn_define_colormap(wks,"wh-bl-gr-ye-re")               ; choose colormap
 
  plot = new(30,graphic)                                    ; create a plot array
   
  fontsz = 0.035
  
  
vNames = (/"NUM_A_ACC", "NUM_A_AIT", "NUM_A_CDU", "NUM_A_CSS", "NUM_A_FDU", "NUM_A_FSS", "NUM_A_PCM"/) 
sNames = (/"ACC", "AIT", "CDU", "CSS", "FDU", "FSS", "PCM"/)

  
;************************************************
; settings
;************************************************
  res                      = True                   ; plot mods desired   
  res@cnFillOn             = True                   ; turn on color
  res@cnLinesOn            = False                  ; no contour lines
  res@cnLineLabelsOn       = False                  ; no contour lines
  ;res@gsnSpreadColors      = True                   ; use full range of colors
  res@lbLabelAutoStride    = True                  ; optimal labels  
 
  res@gsnDraw              = False                   ; don't draw yet
  res@gsnFrame             = False                   ; don't advance frame yet
  
  res@tiMainFontHeightF    = 16
  res@tiMainFont           = "Helvetica-Bold"
  res@lbLabelBarOn          = False
  
  ;res@tmXBLabelsOn         = False              ; do not draw bottom labels
  ;res@tmXBOn               = False              ; no bottom tickmarks
  res@tmXBLabelFontHeightF = fontsz                 ; resize tick labels
  res@tmYLLabelFontHeightF = fontsz  
  res@gsnStringFontHeightF  =fontsz
  
  ;res@mpCenterLonF          = 180    
  res@cnInfoLabelOn = False                     ; turn off cn info label
  ;res@gsnMajorLatSpacing = 30
  ;res@gsnMajorLonSpacing = 60 
 
  res@lbLabelFontHeightF =fontsz
  ;res@pmLabelBarOrthogonalPosF = .2
  ;res@lbBoxEndCapStyle = "TriangleBothEnds"
  res@cnFillMode = "CellFill"
  ;res@tmXBLabelFontHeightF = 0.03                 ; resize tick labels
  ;res@tmYLLabelFontHeightF = 0.03    
  ;res@gsnStringFontHeightF  =0.035
  ;res@labelBarOn = False
 
  resP            = True                         ; modify the panel plot
        
  resP@gsnPanelMainString = ""
  resP@gsnMaximize = True
  resP@gsnFrame    = False
  resP@gsnPanelRight  =  0.98  ; decrease to create more space on the rhs
  resP@gsnPanelLeft  =  0.02  ; increase to create more space on the lhs
  resP@gsnPanelBottom = 0.11  ; increase to create more space
  resP@gsnPanelXWhiteSpacePercent = 3.
  ;resP@gsnPanelYWhiteSpacePercent = 25.
  resP@gsnPanelScalePlotIndex = 1  ; set the size of each panel based on the second plot
      
   
;****************************************************************************
; Set up labelbar resources for vertical and horizontal labelbars
; Some of these may change depending upon the nrow and ncols 
;****************************************************************************
  lbres                     = True                ; common resources for both
                                                  ; horizontal and vertical bars
  lbres@lbPerimOn           = False               ; no label bar box
  lbres@lbLabelFontHeightF  = 0.01                ; label font height
  lbres@lbLabelAlignment    = "InteriorEdges"     ; where to label
  lbres@lbMonoFillPattern   = True                ; solid fill
  lbres@lbBoxEndCapStyle = "TriangleBothEnds"

; ACC Bottom Horizontal Label bar (Projection)
  lbhres_ACC                    = lbres
  lbhres_ACC@lbOrientation      = "Horizontal"        ; orientation
  lbhres_ACC@vpXF               = 0.27  ; increse shifts right
  lbhres_ACC@vpYF               = 0.895  ;0.05  ; increase shifts up
  lbhres_ACC@vpWidthF           = 0.30                ; size
  lbhres_ACC@vpHeightF          = 0.05

; AIT Bottom Horizontal Label bar (Projection)
  lbhres_AIT                    = lbres
  lbhres_AIT@lbOrientation      = "Horizontal"        ; orientation
  lbhres_AIT@vpXF               = 0.27  ;0.18  ; increse shifts right
  lbhres_AIT@vpYF               = 0.765  ;0.05  ; increase shifts up
  lbhres_AIT@vpWidthF           = 0.30                ; size
  lbhres_AIT@vpHeightF          = 0.05

; CDU Bottom Horizontal Label bar (Projection)
  lbhres_CDU                    = lbres
  lbhres_CDU@lbOrientation      = "Horizontal"        ; orientation
  lbhres_CDU@vpXF               = 0.27  ; increse shifts right
  lbhres_CDU@vpYF               = 0.635  ;0.05  ; increase shifts up
  lbhres_CDU@vpWidthF           = 0.30                ; size
  lbhres_CDU@vpHeightF          = 0.05

; CSS Bottom Horizontal Label bar (Projection)
  lbhres_CSS                    = lbres
  lbhres_CSS@lbOrientation      = "Horizontal"        ; orientation
  lbhres_CSS@vpXF               = 0.27  ;0.18  ; increse shifts right
  lbhres_CSS@vpYF               = 0.50  ;0.05  ; increase shifts up
  lbhres_CSS@vpWidthF           = 0.30                ; size
  lbhres_CSS@vpHeightF          = 0.05

; FDU Bottom Horizontal Label bar (Projection)
  lbhres_FDU                    = lbres
  lbhres_FDU@lbOrientation      = "Horizontal"        ; orientation
  lbhres_FDU@vpXF               = 0.27  ;0.18  ; increse shifts right
  lbhres_FDU@vpYF               = 0.37  ;0.05  ; increase shifts up
  lbhres_FDU@vpWidthF           = 0.30                ; size
  lbhres_FDU@vpHeightF          = 0.05

; FSS Bottom Horizontal Label bar (Projection)
  lbhres_FSS                    = lbres
  lbhres_FSS@lbOrientation      = "Horizontal"        ; orientation
  lbhres_FSS@vpXF               = 0.27  ;0.18  ; increse shifts right
  lbhres_FSS@vpYF               = 0.24  ;0.05  ; increase shifts up
  lbhres_FSS@vpWidthF           = 0.30                ; size
  lbhres_FSS@vpHeightF          = 0.05

; PCM Bottom Horizontal Label bar (Projection)
  lbhres_PCM                    = lbres
  lbhres_PCM@lbOrientation      = "Horizontal"        ; orientation
  lbhres_PCM@vpXF               = 0.17  ;0.18  ; increse shifts right
  lbhres_PCM@vpYF               = 0.07  ;0.05  ; increase shifts up
  lbhres_PCM@vpWidthF           = 0.40                ; size
  lbhres_PCM@vpHeightF          = 0.05
  lbhres_PCM@lbTitleString            = "log10(#/kg_air)"
  lbhres_PCM@lbTitleFontHeightF = 0.02
  lbhres_PCM@lbTitlePosition = "Bottom"

; Right Vertical Label bar (Hist)
  lbvres                    = lbres
  lbvres@lbOrientation      = "Vertical"          ; orientation
  lbvres@vpXF               = 0.88  ; increse shifts right   
  lbvres@vpYF               = 0.9
  lbvres@vpWidthF           = 0.05                ; size
  lbvres@vpHeightF          = 0.75
  ;lbvres@lbTitleString            = "log10(MAMnet_prior) - log10(GEOS+MAM)"
  ;lbvres@lbTitlePosition = "Right"
  ;lbvres@lbTitleAngleF = 90.          
   
; read in and plot
;************************************************  
  
  
  data = addfile(mod_name + ".nc","r")
;  vNames = getfilevarnames (data) ; get names of all variables on file
  nNames = dimsizes (vNames)
  lat   = data->lat
  lon   = data->lon
  rad    = 4.0*atan(1.0)/180.0
  clat   = cos(lat*rad)
            
  tit =  "NCONC_zonal_" + tit2 
  wks   = gsn_open_wks (wks_type, tit)                  ; open workstation
  gsn_define_colormap(wks,colormp)
  
 
  print(vNames)
  
  ; define a plot index
  p_idx = 0
  
  do i =  0, nNames-1
  
        var =  vNames(i) 
        print(var)
        ;print(p_idx)
        y_true = data->$var$ (:, :, :, :) 
        var =  var + "_pred"
        y_predx = data->$var$ (:, :, :, :)
        var = sNames(i)
           

        y_true!0 = "time"
        y_true!1 = "lev"
        y_true!2 = "lat"  
        y_true!3 = "lon"
        y_true&lev@units = "hPa"                                    
        y_true&lat@units = "degrees north" 
        y_true&lon@units=  "degrees east"
        y_true@FillValue =  1e+15
        y_pred  = y_true
        y_diff =  y_true
        y_pred  = y_predx

        ;minN =  -10.
        ;y_true  = where( y_true .gt. minN, y_true, y_true@_FillValue)
        ;y_pred  = where( y_pred .gt. minN, y_pred, y_pred@_FillValue)
        
        ; log10(1/mg) -> log10(1/kg)
        y_true = y_true + 6.
        y_pred = y_pred + 6.
        
        ; to use a common label bar, standardize the data and set the contour limits
        nmin = 2. ;0.
        nmax = 7.4 ;1.
        
        ;y_true = (y_true - min(y_true))/(max(y_true)-min(y_true))
        ;y_pred = (y_pred - min(y_pred))/(max(y_pred)-min(y_pred))
        
        
        res@cnFillPalette = colormp
        
         
        ;res@cnLevelSelectionMode = "ManualLevels"
        ;res@gsnLeftString      =  ""          
        ;res@cnMinLevelValF       = nmin        
        ;res@cnMaxLevelValF       = nmax        
        ;res@cnLevelSpacingF      = 0.2; (res@cnMaxLevelValF-  res@cnMinLevelValF)/20.0;
        
        
        res@cnLevelSelectionMode = "ExplicitLevels"
        ;clev = (/2., 2.2, 2.4, 2.6, 2.8, 3., 3.2, 3.4, 3.6, 3.8, 4., 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5., 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6., 6.2, 6.4, 6.6, 6.8, 7., 7.4/)
        
        l_h = ispan (48, 94, 2)*0.01    
        l_h =  10.0^l_h 
        res@cnLevels = l_h
        
        ; calc diff
        y_diff =  y_true
        y_diff  = y_pred-y_true
        
        ; additional plotting options
        res@vpWidthF             = 0.79                    ; change aspect ratio of plot
        res@vpHeightF            = 0.37 
        res@trYReverse            = True
        
        ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
        ; *** true  zonal
        ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

        res@tmYLLabelsOn          = True    ; turn off lat labels
        res@tmXBLabelsOn          = False   ; turn off lon labels
        if (i.eq.6) then
            res@tmXBLabelsOn          = True
        end if

        uavg=y_true(time|0, lev|:, lat|:, lon|:)
        uavg=dim_avg( y_true(lev|:, lat|:, lon|:, time|:) )
        uavg2=uavg(lon|0, lev|:, lat|:)
        uavg2=dim_avg( uavg(lev|:, lat|:, lon|:) )
        ;printVarSummary(uavg2)
        print(min(uavg2))
        print(max(uavg2))
        print(avg(uavg2))
        
        res@gsnLeftString = "GEOS+MAM"
        if(i.gt.0)then
            res@gsnLeftString = ""
        end if
        res@gsnRightString =  "mean = " + sprintf("%1.3f", avg(uavg2(:,:))) 
        plot(p_idx) = gsn_csm_contour(wks,uavg2(:, :), res)  ; create plot 
        
        ;********************************
        ; Horizontal labelbar: Extract info needed for horizontal labelbar
        ;********************************
        if (i.eq.6) then
            lbhid = createLabelBar(wks, plot(p_idx), lbhres_PCM)
        end if
        
        p_idx = p_idx + 1
        
        ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
        ; *** predicted  zonal
        ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
        res@tmYLLabelsOn          = False
        
        uavg=y_pred(time|0, lev|:, lat|:, lon|:)
        uavg=dim_avg( y_pred(lev|:, lat|:, lon|:, time|:) )
        uavg2=uavg(lon|0, lev|:, lat|:)
        uavg2=dim_avg( uavg(lev|:, lat|:, lon|:) )
        
        res@gsnLeftString = "MAMnet"
        if(i.gt.0)then
            res@gsnLeftString = ""
        end if
        res@gsnRightString =  "mean = " + sprintf("%1.3f", avg(uavg2(:,:)))   
        plot(p_idx) = gsn_csm_contour(wks,uavg2(:, :), res)  ; create plot  
        
        p_idx = p_idx + 1

        ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
        ; *** diff
        ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

        res@cnFillPalette = "BlueWhiteOrangeRed"
        res@cnLevelSelectionMode = "ManualLevels"
        res@gsnLeftString      =  ""          
        res@cnMinLevelValF       = -0.5            
        res@cnMaxLevelValF       =  0.5           
        res@cnLevelSpacingF      = 0.05 ;(res@cnMaxLevelValF-  res@cnMinLevelValF)/20.0;
        
        uavg=y_diff(time|0, lev|:, lat|:, lon|:)
        uavg=dim_avg( y_diff(lev|:, lat|:, lon|:, time|:) )
        uavg2=uavg(lon|0, lev|:, lat|:)
        uavg2=dim_avg( uavg(lev|:, lat|:, lon|:) )
        
        res@gsnLeftString = "MLB"
        if(i.gt.0)then
            res@gsnLeftString = ""
        end if
        res@gsnRightString =  "mean = " + sprintf("%1.3f", avg(uavg2(:,:)))  
        plot(p_idx) = gsn_csm_contour(wks,uavg2(:, :), res)  ; create plot 
  
        ;********************************
        ; Vertical labelbar: Extract info needed for horizontal labelbar
        ;********************************
        if (i.eq.0) then
            lbvid = createLabelBar(wks, plot(p_idx), lbvres)
        end if
        
        p_idx = p_idx + 1

        delete(uavg2)
        delete(uavg)  
         
        delete(res@cnLevels)
 
  end do

  ;***********************************************
  ; create combined plot
  ;************************************************


      
  gsn_panel(wks,plot,(/7, 3/),resP)             ; now draw as one plot
  draw(lbhid)                             ; horizontal label bar
  draw(lbvid)                             ; vertical label bar
  ;draw(lbhid_ACC)                             ; horizontal label bar 
  ;draw(lbhid_AIT) 
  ;draw(lbhid_CDU) 
  ;draw(lbhid_CSS) 
  ;draw(lbhid_FDU)  
  ;draw(lbhid_FSS)  
  ;draw(lbhid_PCM)     
  ;frame(wks)  ; the custom label bars will not appear unless you do this


; Draw a text string 
  txres               = True
  txres@txFontHeightF = 0.02           
  txres@txAngleF      = 90.          ; Rotate string clockwise 
  gsn_text_ndc(wks,"Pressure (hPa)",0.02,0.55,txres)
  
  
  txres2               = True
  txres2@txFontHeightF = 0.01 
  ; shifts = (/"[-1]", "[-3]", "[+2]", "[+1]", "[+2]", "[+0]", "[+0]"/) 
  gsn_text_ndc(wks,"ACC",0.07,0.93,txres2)
  gsn_text_ndc(wks,"AIT",0.07,0.8,txres2)
  gsn_text_ndc(wks,"CDU",0.07,0.67,txres2)
  gsn_text_ndc(wks,"CSS",0.07,0.55,txres2)
  gsn_text_ndc(wks,"FDU",0.07,0.42,txres2)
  gsn_text_ndc(wks,"FSS",0.07,0.29,txres2)
  gsn_text_ndc(wks,"PCM",0.07,0.17,txres2)
  
  
  
  
  
  
  frame(wks)
  
        
  destroy(wks)

end
