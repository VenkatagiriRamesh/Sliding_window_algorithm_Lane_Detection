import cv2
import numpy as np
import matplotlib.pyplot as plt
from Line import Line
import glob
import pickle


window_size = 5
left_line = Line(n=window_size)
right_line = Line(n=window_size)
detected = False
left_curve, right_curve = 0., 0.
left_lane_inds, right_lane_inds = None, None


with open('Calibration/cal_pickle.p', mode='rb') as f:
    file = pickle.load(f)
mtx = file['mtx']
dist = file['dist']

cap = cv2.VideoCapture('test_video.mp4')

while(cap.isOpened()):

    retu, frame = cap.read()
    undist = cv2.undistort(frame, mtx, dist, None, mtx)

    img_size = (undist.shape[1], undist.shape[0])
    src = np.float32([[300, 360],[1140, 360],[80, 720],[1280, 720]])
    dst = np.float32([[0, 0],[1280, 0],[0, 720],[1280, 720]])

    m = cv2.getPerspectiveTransform(src, dst)
    m_inv = cv2.getPerspectiveTransform(dst, src)

    binary_warped = cv2.warpPerspective(undist, m, img_size, flags=cv2.INTER_LINEAR)
    binary_unwarped = cv2.warpPerspective(binary_warped, m_inv, (binary_warped.shape[1], binary_warped.shape[0]), flags=cv2.INTER_LINEAR)

    gray = cv2.cvtColor(binary_warped, cv2.COLOR_BGR2GRAY)
    _, b_img = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)

    histogram = np.sum(b_img[b_img.shape[0]//2:,:], axis=0)
    out_img = binary_warped
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 9

    window_height = np.int(b_img.shape[0]/nwindows)


    nonzero = b_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base

    margin = 100

    minpix = 50

    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):

	win_y_low = b_img.shape[0] - (window+1)*window_height
	win_y_high = b_img.shape[0] - window*window_height
	win_xleft_low = leftx_current - margin
	win_xleft_high = leftx_current + margin
	win_xright_low = rightx_current - margin -10
	win_xright_high = rightx_current + margin -10

	cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
	cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)

	good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
	good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

	left_lane_inds.append(good_left_inds)
	right_lane_inds.append(good_right_inds)

	if len(good_left_inds) > minpix:
	    leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
	if len(good_right_inds) > minpix:
	    rightx_current = np.int(np.mean(nonzerox[good_right_inds]))


    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)


    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    if not len(leftx) is 0:
        left_fit = np.polyfit(lefty, leftx, 2)
        lc,res,_,_,_ = np.polyfit(lefty,leftx,2,full = True)
        print('Left lane coefficients : ')
        print(lc)
        lfit = np.poly1d(lc)

    if not len(rightx) is 0:
        right_fit = np.polyfit(righty, rightx, 2)
        rc,res,_,_,_ = np.polyfit(righty,rightx,2,full = True)
        print('Right lane coefficients : ')
        print(rc)
        rfit = np.poly1d(rc)
     
    out_img=cv2.resize(out_img,(640,360))
    cv2.imshow('out_img', out_img)


    if not len(leftx) is 0:
        left_fit = left_line.add_fit(left_fit)
    if not len(rightx) is 0:
        right_fit = right_line.add_fit(right_fit)


    if not len(leftx) is 0:
        y_eval = 719
        ym_per_pix = 5/720
        xm_per_pix = 0.3/1250
        left_curverad = ((1 + (2*left_fit[0]*y_eval*ym_per_pix + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
        right_curverad = ((1 + (2*right_fit[0]*y_eval*ym_per_pix + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])



    if not len(leftx) is 0:
	bottom_y = undist.shape[0] - 1
	bottom_x_left = left_fit[0]*(bottom_y**2) + left_fit[1]*bottom_y + left_fit[2]
	bottom_x_right = right_fit[0]*(bottom_y**2) + right_fit[1]*bottom_y + right_fit[2]
	vehicle_offset = undist.shape[1]/2 - (bottom_x_left + bottom_x_right)/2
	lane_width = (bottom_x_right - bottom_x_left)/2

	#xm_per_pix = 0.3/700
	#vehicle_offset *= xm_per_pix
	vehicle_offset = vehicle_offset*100/lane_width




    ploty = np.linspace(0, undist.shape[0]-1, undist.shape[0])

    if not len(leftx) is 0:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
	right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]


	color_warp = np.zeros((720, 1280, 3), dtype='uint8')


	pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
	pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
	pts = np.hstack((pts_left, pts_right))


	cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))


	newwarp = cv2.warpPerspective(color_warp, m_inv, (undist.shape[1], undist.shape[0]))


	result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
	
	avg_curve = (left_curverad + right_curverad)/2
	label_str = 'Radius of curvature: %.1f cm' % avg_curve
	result = cv2.putText(result, label_str, (30,40), 0, 1, (255,0,0), 2, cv2.LINE_AA)
	
	print('Radius of Curvature : ')
	print(avg_curve)
	print('Vehicle Offset : ')
	print(vehicle_offset)
	label_str = 'Vehicle offset from lane center: %.1f %%' % vehicle_offset
	result = cv2.putText(result, label_str, (30,70), 0, 1, (255,0,0), 2, cv2.LINE_AA)

    	result = cv2.resize(result,(640,360))
   	cv2.imshow('result',result)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# clear everything once finished
cap.release()
cv2.destroyAllWindows()
