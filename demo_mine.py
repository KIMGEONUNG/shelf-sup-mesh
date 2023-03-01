# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function

import os
import numpy as np
import torch
from PIL import Image
import glob

from models.generator import ReconstructModel
from models.evaluator import Evaluator
from nnutils import mesh_utils
from nnutils.utils import load_my_state_dict
import pytorch3d.io

from absl import app
from config.config_flag import *
import argparse

flags.DEFINE_string("demo_image", "examples/demo_images/", "path to input")
flags.DEFINE_string("demo_out", "outputs", "dir of output")
flags.DEFINE_string("ckpt_dir", "weights", "dir of output")
flags.DEFINE_string("path", "input/Teddy-bear/alt1_resize_224.png",
                    "target image")

FLAGS = flags.FLAGS

# optimization lambda
FLAGS.lap_loss = 100
FLAGS.lap_norm_loss = .5
FLAGS.cyc_mask_loss = 10
FLAGS.cyc_perc_loss = 0


def main(_):
    # args = parse()
    data = load_image(FLAGS.path)
    cls = FLAGS.path.split('/')[-2]
    data['cls'] = cls
    prefix = FLAGS.path.split('/')[-1].split('.')[0]
    FLAGS.demo_out = os.path.join(FLAGS.demo_out, cls)

    # load pretrianed model
    try:
        model, cfg = load_model(os.path.join(FLAGS.ckpt_dir, cls, 'model.pth'))
    except FileNotFoundError:
        print(os.path.join(FLAGS.ckpt_dir, cls, 'model.pth'))

    # for visualization utils
    evaluator = Evaluator(cfg)

    # step1: infer coarse shape and camera pose
    vox_world, camera_param = model.forward_image(data['image'])
    # init meshes
    vox_mesh = mesh_utils.cubify(vox_world).clone()
    # step2: optimize meshes
    mesh_inputs = {'mesh': vox_mesh, 'view': camera_param}
    with torch.enable_grad():
        mesh_outputs, record = evaluator.opt_mask(model, mesh_inputs, data,
                                                  True, 300)
    # visualize mesh.
    vis_mesh(mesh_outputs, camera_param, evaluator.snapshot_mesh, prefix)

    # save mesh
    mesh = mesh_outputs['mesh']
    path_mesh = os.path.join(FLAGS.demo_out, f"{prefix}_mesh.obj")
    pytorch3d.io.save_obj(path_mesh, mesh.verts_packed(), mesh.faces_packed())
    print('Program finished ')


def load_image(path_img):
    image = Image.open(path_img)
    assert image.mode == "RGBA"
    image = np.asarray(image)

    # SPLIT IMAGE AND MASK
    mask = image[:, :, 3:]  # in case of RGBA
    mask = np.concatenate([mask, mask, mask], axis=-1)
    image = image[:, :, :3]  # in case of RGBA

    # RENORM IMAGE
    image = image / 127.5 - 1  # [-1, 1]

    mask = (mask > 0).astype(np.float)
    fg = image * mask + (1 - mask)  # white bg

    fg = to_tensor(fg)
    image = to_tensor(image)
    mask = to_tensor(mask)

    return {'bg': image, 'image': fg, 'mask': mask}


def to_tensor(image):
    image = np.transpose(image, [2, 0, 1])
    image = image[np.newaxis]
    return torch.FloatTensor(image).cuda()


def load_model(ckpt_file):
    print('Init...', ckpt_file)
    pretrained_dict = torch.load(ckpt_file)
    cfg = pretrained_dict['cfg']

    model = ReconstructModel()
    load_my_state_dict(model, pretrained_dict['G'])

    model.eval()
    model.cuda()
    return model, cfg


def vis_mesh(cano_mesh, pred_view, snapshot_func, prefix, f=375):
    """
    :param cano_mesh:
    :param pred_view:
    :param renderer:
    :param snapshot_func: snapshot given pose_list, and generate gif.
    :return:
    """
    # a novel view
    snapshot_func(cano_mesh['mesh'][-1], [],
                  None,
                  FLAGS.demo_out,
                  prefix,
                  'mesh',
                  pred_view=pred_view)
    snapshot_func(cano_mesh['mesh'][-1], [],
                  cano_mesh['mesh'].textures,
                  FLAGS.demo_out,
                  prefix,
                  'meshTexture',
                  pred_view=pred_view)


if __name__ == '__main__':
    app.run(main)
