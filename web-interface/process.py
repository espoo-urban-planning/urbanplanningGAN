import sys
import vec
import thin
import graph

save_path = '/usr/local/Cellar/nginx/1.15.2/html/n/tmp/'


basename = sys.argv[1]
first_input = save_path + basename
task_id = basename.replace(".png", "")

vec.convert_to_vector(first_input, save_path + task_id + "_0.png")
thin.thin_image(save_path + task_id + "_0.png", save_path + task_id + "_1.png")
graph.create_dxf(save_path + task_id + "_1.png", save_path + task_id + '.dxf')

sys.exit(0)
