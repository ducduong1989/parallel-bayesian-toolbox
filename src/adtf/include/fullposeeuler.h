#ifndef FULLPOSEEULER_H
#define FULLPOSEEULER_H

#define MEDIA_TYPE_FULLPOSEEULER	0x00050001  //TODO: a versioned version would be better
#define MEDIA_SUBTYPE_EULER_XYZ		0x00051001  //defines rotation order

//rotation matrix from a given pose, given through Euler angles:
	// Rot(0,0) = cosY*cosZ;
	// Rot(0,1) = cosZ*sinX*sinY-cosX*sinZ;
	// Rot(0,2) = cosX*cosZ*sinY+sinX*sinZ;
	// Rot(1,0) = cosY*sinZ;
	// Rot(1,1) = cosX*cosZ+sinX*sinY*sinZ;
	// Rot(1,2) = cosX*sinY*sinZ-cosZ*sinX;
	// Rot(2,0) = -sinY;
	// Rot(2,1) = cosY*sinX;
	// Rot(2,2) = cosX*cosY;	
//where cos# is the cosinus of the Euler angle at the 

//translation after rotation (in target coordinate system)

//these transformation should generate a 3x4 matrix with the first 3 columns being the 
//rotation matrix, while the last column is the translation vector. The matrix can be
//applied to transform 3D homogeneous points (4x1) between two coordinate systems.
	
struct FullPoseEuler
{
		//translation in x direction (m)
		float tx;
		//translation in y direction (m)
		float ty;
		//translation in z direction (m)
		float tz;
		//rotation in rad around x (3rd to rotate)
		float rx; 
		//rotation in rad around y (2nd to rotate)
		float ry;
		//rotation in rad around z (1st to rotate)
		float rz; //rotates first
};


#endif // FULLPOSEEULER_H