import matplotlib.pyplot as plt
from micarrays import Eigenmike64

em = Eigenmike64()

# Get Cartesian coordinates: shape (64, 3)
coords = em.coordinates_cartesian
x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

# Plot 3D scatter
fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection="3d")

ax.scatter(x, y, z, s=40, edgecolor="k")
ax.set_title("Eigenmike64 Microphone Positions (Cartesian)")
ax.set_xlabel("X (meters)")
ax.set_ylabel("Y (meters)")
ax.set_zlabel("Z (meters)")
ax.set_box_aspect([1, 1, 1])

plt.tight_layout()
plt.savefig("eigenmike64_cartesian.png", dpi=300, bbox_inches="tight")
plt.close()
