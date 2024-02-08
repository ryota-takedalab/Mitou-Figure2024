import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.art3d as art3d
import numpy as np

from .bone import H36M_BONE, WHOLE_BONE


def setLines(X, Y, Z):
    lineX = []
    lineY = []
    lineZ = []

    for bone in H36M_BONE:
        lineX.append([X[bone[0]], X[bone[1]]])
        lineY.append([Y[bone[0]], Y[bone[1]]])
        lineZ.append([Z[bone[0]], Z[bone[1]]])

    return np.array(lineX), np.array(lineY), np.array(lineZ)


def setLinesWhole(X, Y, Z):
    lineX = []
    lineY = []
    lineZ = []

    for bone in WHOLE_BONE:
        lineX.append([X[bone[0]], X[bone[1]]])
        lineY.append([Y[bone[0]], Y[bone[1]]])
        lineZ.append([Z[bone[0]], Z[bone[1]]])

    return np.array(lineX), np.array(lineY), np.array(lineZ)


def vis_calib_res(kpts3d_est, kpts3d):
    fig = plt.figure()
    ax0 = fig.add_subplot(232, projection="3d")
    ax1 = fig.add_subplot(234, projection="3d")
    ax2 = fig.add_subplot(235, projection="3d")
    ax3 = fig.add_subplot(236, projection="3d")

    def draw_skeleton(ax, X, Y, Z, title):
        ax.clear()
        ax.set_title(title)
        ax.view_init(elev=-90, azim=-86)
        if "Calibrated" in title:
            ax.set_xlim(np.nanmin(kpts3d_est[:, :, 0]), np.nanmax(kpts3d_est[:, :, 0]))
            ax.set_ylim(np.nanmin(kpts3d_est[:, :, 1]), np.nanmax(kpts3d_est[:, :, 1]))
            ax.set_zlim(np.nanmin(kpts3d_est[:, :, 2]), np.nanmax(kpts3d_est[:, :, 2]))
        elif "Cam 1" in title:
            ax.set_xlim(np.nanmin(kpts3d[0, :, :, 0]), np.nanmax(kpts3d[0, :, :, 0]))
            ax.set_ylim(np.nanmin(kpts3d[0, :, :, 1]), np.nanmax(kpts3d[0, :, :, 1]))
            ax.set_zlim(np.nanmin(kpts3d[0, :, :, 2]), np.nanmax(kpts3d[0, :, :, 2]))
        elif "Cam 2" in title:
            ax.set_xlim(np.nanmin(kpts3d[1, :, :, 0]), np.nanmax(kpts3d[1, :, :, 0]))
            ax.set_ylim(np.nanmin(kpts3d[1, :, :, 1]), np.nanmax(kpts3d[1, :, :, 1]))
            ax.set_zlim(np.nanmin(kpts3d[1, :, :, 2]), np.nanmax(kpts3d[1, :, :, 2]))
        elif "Cam 3" in title:
            ax.set_xlim(np.nanmin(kpts3d[2, :, :, 0]), np.nanmax(kpts3d[2, :, :, 0]))
            ax.set_ylim(np.nanmin(kpts3d[2, :, :, 1]), np.nanmax(kpts3d[2, :, :, 1]))
            ax.set_zlim(np.nanmin(kpts3d[2, :, :, 2]), np.nanmax(kpts3d[2, :, :, 2]))

        ax.plot(X, Y, Z, "k.")

        if "Calibrated" in title:
            X_bone, Y_bone, Z_bone = setLinesWhole(X, Y, Z)
        else:
            X_bone, Y_bone, Z_bone = setLines(X, Y, Z)
        for x, y, z in zip(X_bone, Y_bone, Z_bone):
            line = art3d.Line3D(x, y, z, color="#f94e3e")
            ax.add_line(line)

    def update_frame(fc):
        X0 = kpts3d_est[fc, :, 0]
        Y0 = kpts3d_est[fc, :, 1]
        Z0 = kpts3d_est[fc, :, 2]
        draw_skeleton(ax0, X0, Y0, Z0, "Calibrated 3D pose")

        X1 = kpts3d[0, fc, :, 0]
        Y1 = kpts3d[0, fc, :, 1]
        Z1 = kpts3d[0, fc, :, 2]
        draw_skeleton(ax1, X1, Y1, Z1, "Cam 1")

        X2 = kpts3d[1, fc, :, 0]
        Y2 = kpts3d[1, fc, :, 1]
        Z2 = kpts3d[1, fc, :, 2]
        draw_skeleton(ax2, X2, Y2, Z2, "Cam 2")

        X3 = kpts3d[2, fc, :, 0]
        Y3 = kpts3d[2, fc, :, 1]
        Z3 = kpts3d[2, fc, :, 2]
        draw_skeleton(ax3, X3, Y3, Z3, "Cam 3")

    ani = animation.FuncAnimation(
        fig, update_frame, frames=kpts3d_est.shape[0], interval=30, repeat=False
    )
    return ani
