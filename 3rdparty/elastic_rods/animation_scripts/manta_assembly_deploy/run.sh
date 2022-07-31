python3 generate_geometry.py
python3 generate_frames.py

pushd frames
for i in {0..299}; do
    gmsh_offscreen -n frame_$i.msh render_$i.opt -o frame_$i.png;
    convert -resize 50% frame_$i.png resized_$i.png;
done
popd

ffmpeg -f image2 \
    -framerate 30 \
    -i frames/resized_%d.png \
    -c:v libx264 \
    -preset veryslow \
    -qp 18 \
    -pix_fmt yuv420p \
    assemble_deploy.mp4
