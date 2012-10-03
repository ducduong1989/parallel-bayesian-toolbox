#ifndef CALIBRATIONSTATE_H
#define CALIBRATIONSTATE_H

#define MEDIA_TYPE_CALIBRATIONSTATE	0x00049001  //TODO: a versioned version would be better
#define MEDIA_SUBTYPE_PINHOLEMODEL	0x00049002  

struct CalibrationState
{
		float cx;	//principal point
		float cy;
		float fx;	//internal focal scale
		float fy;
		float tx; //external translation
		float ty;
		float tz;		
		float r1; //external rotation in Rodrigues representation
		float r2;
		float r3;

		//coordinates of edges of mapped/unmapped area in birdeye view
		//representaion into 2 lines (bottom line can be achieved by bottom left+right points)
		//line mapping the image left edge
		float blx; //bottom left x coordinate
		float bly; //bottom left y coordinate
		float ldx; //left edge direction x
		float ldy; //left edge direction y

		//line mapping the image right edge
		float brx; //bottom right x coordinate
		float bry; //bottom right y coordinate
		float rdx; //right edge direction x
		float rdy; //right edge direction y
};


#endif // CALIBRATIONSTATE_H