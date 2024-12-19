




;*************************************************


;************************************************
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl" 
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
;load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"   
;load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/shea_util.ncl"
;************************************************
;----------------------------------------------------------------------
; from panel_29.ncl
;----------------------------------------------------------------------
;     Local function that extracts information from the
;     contour object attribute associated with the input plot. 
;     This makes the code a bit cleaner also.  
;----------------------------------------------------------------------
undef("createLabelBar")
function createLabelBar(wks: graphic, plt:graphic, lbres:logical)
begin
   getvalues plt@contour               ; object from which info extracted
      "cnFillColors"   : fill_colors   ; get the colors used
      "lbLabelStrings" : lbstrings     ; get the labels used
   end getvalues
   
   delete(lbres@lbFillColors)
   lbres@lbFillColors = fill_colors    ; associate with input lnres argument
   lbid = gsn_create_labelbar(wks,dimsizes(lbstrings)+1,lbstrings,lbres)
   return(lbid)
end

begin
 

;;;;;"net" is MASS or NUM, from the command line

   Nx =  37
   pres=1

   levplot=  (/71, 55, 43/)
   tit2 = "_diff_commonlbar"
   mod_name = "MAMnet_test"
   dlev =  1
   if (pres .gt. 0) then
     delete(levplot)
     delete(dlev)
     levplot=  (/950.0, 500.0, 250.0/)
     tit2 = "_diff_pres_commonlbar"
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
   
  fontsz = 0.040
  
  
vNames = (/"DPG_ACC", "DPG_AIT", "DPG_CDU", "DPG_CSS", "DPG_FDU", "DPG_FSS", "DPG_PCM"/) 
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
  
  res@mpCenterLonF          = 180    
  res@cnInfoLabelOn = False                     ; turn off cn info label
  res@gsnMajorLatSpacing = 30
  res@gsnMajorLonSpacing = 60 
 
  res@lbLabelFontHeightF =fontsz
  ;res@pmLabelBarOrthogonalPosF = .2
  ;res@lbBoxEndCapStyle = "TriangleBothEnds"
  res@cnFillMode = "CellFill"
  ;res@tmXBLabelFontHeightF = 0.03                 ; resize tick labels
  ;res@tmYLLabelFontHeightF = 0.03    
  ;res@gsnStringFontHeightF  =0.035
 
  resP            = True                         ; modify the panel plot
        
  resP@gsnPanelMainString = ""
  resP@gsnMaximize = True
  resP@gsnFrame    = False
  resP@gsnPanelRight  =  0.98  ; decrease to create more space on the rhs
  resP@gsnPanelLeft  =  0.01  ; increase to create more space on the lhs
  resP@gsnPanelBottom = 0.11  ; increase to create more space
  ;resP@gsnPanelXWhiteSpacePercent = 3.
  ;resP@gsnPanelYWhiteSpacePercent = 25.
  resP@gsnPanelScalePlotIndex = 1  ; set the size of each panel based on the second plot
      
   
;****************************************************************************
; Set up labelbar resources for vertical and horizontal labelbars
; Some of these may change depending upon the nrow and ncols 
;****************************************************************************
  lbres                     = True                ; common resources for both
                                                  ; horizontal and vertical bars
  lbres@lbPerimOn           = False               ; no label bar box
  lbres@lbLabelFontHeightF  = 0.008                ; label font height
  lbres@lbLabelAlignment    = "InteriorEdges"     ; where to label
  lbres@lbMonoFillPattern   = True                ; solid fill
  lbres@lbBoxEndCapStyle = "TriangleBothEnds"

; PCM Bottom Horizontal Label bar (Projection)
  lbhres_PCM                    = lbres
  lbhres_PCM@lbOrientation      = "Horizontal"        ; orientation
  lbhres_PCM@vpXF               = 0.22  ; increse shifts right
  lbhres_PCM@vpYF               = 0.08  ;0.05  ; increase shifts up
  lbhres_PCM@vpWidthF           = 0.30                ; size
  lbhres_PCM@vpHeightF          = 0.05
  lbhres_PCM@lbTitleString            = "Dpg (microns)"
  lbhres_PCM@lbTitlePosition = "Bottom"

; Right Vertical Label bar (Hist)
  lbvres                    = lbres
  lbvres@lbOrientation      = "Vertical"          ; orientation
  lbvres@vpXF               = 0.84  ; increse shifts right   
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
            
  tit =  "DPG_surf" + tit2 
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
        
        res@cnFillPalette = colormp
        delete(res@cnLevelSpacingF)  

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
        
        y_diff =  y_true
        y_diff  = y_pred-y_true
        
        
        ; log10(m) -> log10(microns)
        ;y_true = y_true + 6.
        ;y_pred = y_pred + 6.
        
        y_true =  10^y_true
        y_pred =  10^y_pred
       
        res@cnFillPalette = colormp
        
        
        res@cnLevelSelectionMode = "ExplicitLevels"
        ;clev = (/4., 4.1, 4.2, 4.3, 4.4, 4.5, 4.55, 4.6, 4.65, 4.7, 4.75, 4.8, 4.85, 4.9, 4.95, 5., 5.05, 5.1, 5.15, 5.2, 5.25, 5.3, 5.35, 5.4, 5.45, 5.5, 5.55, 5.6, 5.65, 5.7, 5.75, 5.8, 5.85, 5.9, 6./)
        ;clev = (/4., 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5., 5.1, 5.2, 5.3, 5.4, 5.5, 6., 6.5/)
        ;clev = (/4., 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5.0, 5.2, 5.4, 5.6, 5.8, 6.0, 6.2, 6.4/)
        ;clev = (/3.8, 4., 4.2, 4.4, 4.6, 4.8, 4.85, 4.9, 5., 5.1, 5.2, 5.3, 5.4, 5.45, 5.5, 5.55, 5.6, 5.7, 5.75, 5.8/)
        
        ;clev = (/3.8, 4., 4.2, 4.4, 4.6, 4.8, 4.85, 4.9, 5., 5.1, 5.2, 5.3, 5.4, 5.45, 5.5, 5.55, 5.6, 5.7/)
        
        l_h = ispan (-220, 35, 15)*0.01    
        l_h =  10.0^l_h 
        print(l_h)
        res@cnLevels = l_h
        
        /;
        nmin = 3. 
        nmax = 5.6 
        res@cnLevelSelectionMode = "ManualLevels"
        res@gsnLeftString      =  ""          
        res@cnMinLevelValF       = nmin        
        res@cnMaxLevelValF       = nmax        
        res@cnLevelSpacingF      = 0.05; (res@cnMaxLevelValF-  res@cnMinLevelValF)/20.0;
        ;/
        
        ; calc diff
        
        
        ; plot surface
       
        lv  = levplot(0) ; up one from surface
        lv1 =  lv-dlev
        lv2 =  lv+dlev
        
        ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
        ; *** true  surface
        ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

        res@tmYLLabelsOn          = True    ; turn off lat labels
        res@tmXBLabelsOn          = False   ; turn off lon labels
        if (i.eq.6) then
            res@tmXBLabelsOn          = True
        end if

        y=y_true(:, {lv1:lv2}, :, :) 
        uavg=y(time|:, lev|0, lat|:, lon|:)
        uavg=dim_avg(y(time|:, lat|:, lon|:, lev|:))        
        uavg2=uavg(time|0, lat|:, lon|:)
        uavg2=dim_avg(uavg( lat|:, lon|:, time|:))
        
        print(min(uavg2))
        print(max(uavg2))
        print(avg(uavg2))
        
        res@gsnLeftString = ""
        if(i.eq.0)then
            res@gsnLeftString = "GEOS+MAM"
        end if
        res@gsnRightString =  "mean = " + sprintf("%1.3f", wgt_areaave(uavg2, clat, 1.0, 0))  
        plot(p_idx) = gsn_csm_contour_map(wks,uavg2, res)  ; create plot 
        p_idx = p_idx + 1
        ;print(p_idx)
  
        ;********************************
        ; Horizontal labelbar: Extract info needed for horizontal labelbar
        ;********************************
        if (i.eq.6) then
            lbhid_PCM = createLabelBar(wks, plot, lbhres_PCM)
        end if

        getvalues plot@contour           ; Retrieve contour levels.
    		"cnLevels"   : clev
  		end getvalues
  
        res@cnLevelSelectionMode = "ExplicitLevels"       
        res@cnLevels =  clev
        
        ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
        ; *** predicted  surface
        ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
        res@tmYLLabelsOn          = False
        
        y=y_pred(:, {lv1:lv2}, :, :) 
        uavg=y(time|:, lev|0, lat|:, lon|:)
        uavg=dim_avg(y(time|:,lat|:, lon|:, lev|:))
        uavg2=uavg(time|0, lat|:, lon|:)
        uavg2=dim_avg(uavg( lat|:, lon|:, time|:)) 
        
        res@gsnLeftString = ""
        if(i.eq.0)then
            res@gsnLeftString = "MAMnet"
        end if
        res@gsnRightString =  "mean = " + sprintf("%1.3f", wgt_areaave_Wrap(uavg2, clat, 1.0, 0))  
        plot(p_idx) = gsn_csm_contour_map(wks,uavg2, res)  ; create plot 
        p_idx = p_idx + 1
        ;print(p_idx)

        ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
        ; *** diff
        ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

        res@cnFillPalette = "BlueWhiteOrangeRed"
        res@cnLevelSelectionMode = "ManualLevels"
        res@gsnLeftString      =  ""          
        res@cnMinLevelValF       = -0.5            
        res@cnMaxLevelValF       =  0.5           
        res@cnLevelSpacingF      = 0.05 ;(res@cnMaxLevelValF-  res@cnMinLevelValF)/20.0;
        
        y=y_diff(:, {lv1:lv2}, :, :) 
        uavg=y(time|:, lev|0, lat|:, lon|:)
        uavg=dim_avg(y(time|:,lat|:, lon|:, lev|:))
        uavg2=uavg(time|0, lat|:, lon|:)
        uavg2=dim_avg(uavg( lat|:, lon|:, time|:))
        
        res@gsnLeftString = ""
        if(i.eq.0)then
            res@gsnLeftString = "MLB"
        end if
        res@gsnRightString =  "mean = " + sprintf("%1.3f", wgt_areaave(uavg2, clat, 1.0, 0))  
        plot(p_idx) = gsn_csm_contour_map(wks,uavg2, res)  ; create plot 
        p_idx = p_idx + 1
        ;print(p_idx)
  
        ;********************************
        ; Vertical labelbar: Extract info needed for horizontal labelbar
        ;********************************
        if (i.eq.0) then
            lbvid = createLabelBar(wks, plot, lbvres)
        end if

        delete(uavg2)
        delete(uavg)  
         
        delete(res@cnLevels)
        delete(clev)
 
  end do

  ;***********************************************
  ; create combined plot
  ;************************************************


      
  gsn_panel(wks,plot,(/7, 3/),resP)             ; now draw as one plot
  draw(lbvid)                             ; vertical label bar
  draw(lbhid_PCM)                             ; horizontal label bar    
  
  
  txres2               = True
  txres2@txFontHeightF = 0.01 
  ; shifts = (/"[+0]", "[+1]", "[-1]", "[+0]", "[+0]", "[+0]", "[+0]"/)
  gsn_text_ndc(wks,"ACC",0.09,0.93,txres2)
  gsn_text_ndc(wks,"AIT",0.09,0.8,txres2)
  gsn_text_ndc(wks,"CDU",0.09,0.67,txres2)
  gsn_text_ndc(wks,"CSS",0.09,0.55,txres2)
  gsn_text_ndc(wks,"FDU",0.09,0.42,txres2)
  gsn_text_ndc(wks,"FSS",0.09,0.29,txres2)
  gsn_text_ndc(wks,"PCM",0.09,0.17,txres2)
  
  frame(wks)  ; the custom label bars will not appear unless you do this
        
  destroy(wks)

end
