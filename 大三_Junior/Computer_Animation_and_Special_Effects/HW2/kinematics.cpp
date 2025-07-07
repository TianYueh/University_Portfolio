#include "simulation/kinematics.h"

#include <iostream>
#include "Eigen/Dense"
#include "acclaim/bone.h"
#include "util/helper.h"

namespace kinematics {

void forwardSolver(const acclaim::Posture& posture, acclaim::Bone* bone) {
    // TODO (FK)
    // You should set these variables:
    //     bone->start_position = Eigen::Vector4d::Zero();
    //     bone->end_position = Eigen::Vector4d::Zero();
    //     bone->rotation = Eigen::Matrix4d::Zero();
    // The sample above just set everything to zero
    // Hint:
    //   1. posture.bone_translations, posture.bone_rotations
    // Note:
    //   1. This function will be called with bone == root bone of the skeleton
    //   2. we use 4D vector to represent 3D vector, so keep the last dimension as "0"
    //   3. util::rotate{Degree | Radian} {XYZ | ZYX}
    //      e.g. rotateDegreeXYZ(x, y, z) means:
    //      x, y and z are presented in degree rotate z degrees along z - axis first, then y degrees along y - axis, and
    //      then x degrees along x - axis

    bone->start_position = Eigen::Vector4d::Zero();
    bone->end_position = Eigen::Vector4d::Zero();
    bone->rotation = Eigen::Matrix4d::Zero();

    Eigen::Quaterniond r = util::rotateDegreeZYX(posture.bone_rotations[bone->idx]);

    if (bone->name == "root") {
		bone->rotation = r.toRotationMatrix();
		bone->start_position = posture.bone_translations[bone->idx];
        bone->end_position = bone->start_position;
    }
    else {
		bone->rotation = bone->parent->rotation * bone->rot_parent_current * r.toRotationMatrix();
        bone->start_position = bone->parent->end_position + posture.bone_translations[bone->idx];
        bone->end_position = bone->start_position + bone->rotation * bone->dir * bone->length;
    }

    if (bone->child != nullptr) {
		forwardSolver(posture, bone->child);
	}
    if (bone->sibling != nullptr) {
        forwardSolver(posture, bone->sibling);
    }


}

Eigen::VectorXd pseudoInverseLinearSolver(const Eigen::Matrix4Xd& Jacobian, const Eigen::Vector4d& target) {
    // TODO (find x which min(| jacobian * x - target |))
    // Hint:
    //   1. Linear algebra - least squares solution
    //   2. https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse#Construction
    // Note:
    //   1. SVD or other pseudo-inverse method is useful
    //   2. Some of them have some limitation, if you use that method you should check it.
    Eigen::VectorXd deltatheta(Jacobian.cols());
    // get pseudo inverse of the Jacobian with SVD
    Eigen::JacobiSVD<Eigen::Matrix4Xd> svd(Jacobian, Eigen::ComputeThinU | Eigen::ComputeThinV);
    deltatheta = svd.solve(target);
    return deltatheta;
}

/**
 * @brief Perform inverse kinematics (IK)
 *
 * @param target_pos The position where `end_bone` (first joint in the chain) will move to.
 * @param posture The original AMC motion's reference, you need to modify this
 * @param jointChains A 2D vector containing multiple 1D vectors, each of which holds pointers to Eigen::Vector4d
 * constituting a chain.
 * @param boneChains A 2D vector containing multiple 1D vectors, each of which holds pointers to acclaim::Bone
 * constituting a chain.
 * @param currentBasePos The base of the current chain.
 */

bool inverseJacobianIKSolver(std::vector<Eigen::Vector4d> target_pos, acclaim::Bone* end_bone,
                             acclaim::Posture& posture, std::vector<std::vector<Eigen::Vector4d*>>& jointChains,
                             std::vector<std::vector<acclaim::Bone*>>& boneChains, Eigen::Vector4d currentBasePos) {
    constexpr int max_iteration = 1000;
    constexpr double epsilon = 1E-3;
    constexpr double step = 0.15;
    // Since bone stores in bones[i] that i == bone->idx, we can use bone - bone->idx to find bones[0] which is the
    // root.
    acclaim::Bone* root_bone = end_bone - end_bone->idx;
    // TODO
    // Perform inverse kinematics (IK)
    // HINTs will tell you what should do in that area.
    // Of course you can ignore it (Any code below this line) and write your own code.
    acclaim::Posture original_posture(posture);

    size_t bone_num = 0;

    // Traverse each chain
    for (int chainIdx = 0; chainIdx < boneChains.size(); ++chainIdx) {
        bone_num = boneChains[chainIdx].size();
        Eigen::Matrix4Xd Jacobian(4, 3 * bone_num);
        Jacobian.setZero();

        for (int iter = 0; iter < max_iteration; ++iter) {
            //forwardSolver(posture, root_bone);
            Eigen::Vector4d desiredVector = target_pos[chainIdx] - *jointChains[chainIdx][0];
            //std::cout<<desiredVector<<std::endl;
            if (desiredVector.norm() < epsilon) {
                break;
            }
            // TODO (compute jacobian)
            //   1. Compute arm vectors
            //   2. Compute jacobian columns, store in `Jacobian`
            // Hint:
            //   1. You should not put rotation in jacobian if it doesn't have that DoF.
            //   2. jacobian.col(/* some column index */) = /* jacobian column */

            
            for (long long i = 0; i < bone_num; i++) {
                Eigen::Vector3d arm =
                    jointChains[chainIdx][0]->head<3>() - jointChains[chainIdx][i+1]->head<3>();
                Eigen::Affine3d rotation = boneChains[chainIdx][i]->rotation;
                if (boneChains[chainIdx][i]->dofrx) {
					Eigen::Vector3d unit_rotation = rotation.matrix().col(0).head<3>();
					Eigen::Vector3d J = unit_rotation.cross(arm);
                    Jacobian.col(3 * i) = Eigen::Vector4d(J[0], J[1], J[2], 0);
				}
				if (boneChains[chainIdx][i]->dofry) {
                    Eigen::Vector3d unit_rotation = rotation.matrix().col(1).head<3>();
					Eigen::Vector3d J = unit_rotation.cross(arm);
					Jacobian.col(3 * i + 1) = Eigen::Vector4d(J[0], J[1], J[2], 0);
				}
				if (boneChains[chainIdx][i]->dofrz) {
                    Eigen::Vector3d unit_rotation = rotation.matrix().col(2).head<3>();
					Eigen::Vector3d J = unit_rotation.cross(arm);
					Jacobian.col(3 * i + 2) = Eigen::Vector4d(J[0], J[1], J[2], 0);

				}
            }


            Eigen::VectorXd deltatheta = step * pseudoInverseLinearSolver(Jacobian, desiredVector);

            // TODO (update rotation)
            //   Update `posture.bone_rotation` (in euler angle / degrees) using deltaTheta
            // Hint:
            //   1. You can ignore rotation limit of the bone.
            // Bonus:
            //   1. You cannot ignore rotation limit of the bone.

            for (long long i = 0; i < bone_num; i++) {
                acclaim::Bone curr = *boneChains[chainIdx][i];
                Eigen::Vector3d delta = deltatheta.segment(i * 3, 3);
                posture.bone_rotations[curr.idx] += util::toDegree(Eigen::Vector4d(delta[0], delta[1], delta[2], 0));
                
                
                if (posture.bone_rotations[curr.idx][0] < curr.rxmin) {
                    posture.bone_rotations[curr.idx][0] = curr.rxmin;
                }
                else if (posture.bone_rotations[curr.idx][0] > curr.rxmax) {
					posture.bone_rotations[curr.idx][0] = curr.rxmax;
				}
                if (posture.bone_rotations[curr.idx][1] < curr.rymin) {
                    posture.bone_rotations[curr.idx][1] = curr.rymin;
                }
                else if (posture.bone_rotations[curr.idx][1] > curr.rymax) {
                    posture.bone_rotations[curr.idx][1] = curr.rymax;
                }
                if (posture.bone_rotations[curr.idx][2] < curr.rzmin) {
					posture.bone_rotations[curr.idx][2] = curr.rzmin;
				}
                else if (posture.bone_rotations[curr.idx][2] > curr.rzmax) {
					posture.bone_rotations[curr.idx][2] = curr.rzmax;
				}
                

            }

            forwardSolver(posture, root_bone);
            // Deal with root translation
            if (chainIdx == 0) {
                posture.bone_translations[0] =
                    posture.bone_translations[0] + (currentBasePos - *jointChains[chainIdx][bone_num]);
            }
        }
    }

    // Return whether IK is stable (i.e. whether the ball is reachable) and let the skeleton not swing its hand in the
    // air
    bool stable = true;
    for (int i = 0; i < boneChains.size(); ++i) {
        if ((target_pos[i] - *jointChains[i][0]).norm() > epsilon) {
            //std::cout << target_pos[i] << std::endl;
            //std::cout << *jointChains[i][0] << std::endl;
            //std::cout << std::endl;
            stable = false;
        }
    }
    // You can replace "!stable" with "false" to see unstable results, but this may lead to some unexpected outcomes.
    if (!stable) {
        posture = original_posture;
        forwardSolver(posture, root_bone);
        return false;
    } else {
        original_posture = posture;
        return true;
    }
}
}  // namespace kinematics


