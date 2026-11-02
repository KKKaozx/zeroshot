"""Read a fixed body-to-site transform from an explicit MuJoCo XML snapshot.

No simulator, rollout, robot calibration fit, or guessed XML orientation syntax.
The observation may combine a site's origin with a different body's axes.
That is a valid virtual frame; its quaternion need not equal controller-site
orientation. Never silently replace the trained policy's orientation convention.
"""
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


def _local_transform(node):
    if any(key in node.attrib for key in ('euler', 'axisangle', 'xyaxes', 'zaxis')):
        raise ValueError('Only explicit quaternion/default-identity XML orientations supported')
    position = np.fromstring(node.get('pos', '0 0 0'), sep=' ')
    wxyz = np.fromstring(node.get('quat', '1 0 0 0'), sep=' ')
    if (position.shape != (3,) or wxyz.shape != (4,) or not np.isfinite(position).all()
            or not np.isfinite(wxyz).all() or np.linalg.norm(wxyz) < 1e-12):
        raise ValueError('Invalid fixed XML transform')
    return position, Rotation.from_quat(wxyz[[1, 2, 3, 0]])


def fixed_body_to_site(xml, *, quaternion_body_name, position_site_name):
    """Return the site origin/axes in a declared ancestor body's local frame.

    Reject a joint on the intervening path: a moving finger site cannot be
    represented as a constant EEF offset. Lengths remain XML model units.
    """
    root = ET.fromstring(xml)
    if root.find('.//include') is not None or any(len(x) for x in root.iter('default')):
        raise ValueError('Expanded XML without inherited defaults required')
    bodies = [x for x in root.iter('body') if x.get('name') == quaternion_body_name]
    sites = [x for x in root.iter('site') if x.get('name') == position_site_name]
    if len(bodies) != 1 or len(sites) != 1:
        raise ValueError('Unique explicit body and site required')
    parents = {child: parent for parent in root.iter() for child in parent}
    body, node, path = bodies[0], sites[0], []
    while node is not body:
        if node.tag not in ('body', 'site'):
            raise ValueError('Site must descend from the declared quaternion body')
        if node.tag == 'body' and (node.find('joint') is not None or node.find('freejoint') is not None):
            raise ValueError('Body-to-site path contains a moving joint')
        path.append(node)
        node = parents.get(node)
        if node is None:
            raise ValueError('Site does not descend from the declared quaternion body')
    position, rotation = np.zeros(3), Rotation.identity()
    for node in reversed(path):
        local_p, local_r = _local_transform(node)
        position += rotation.apply(local_p)
        rotation = rotation * local_r
    return {'format': 'libero-eef-observation-frame-definition-v1',
        'position_site_name': position_site_name, 'quaternion_body_name': quaternion_body_name,
        'position_site_in_quaternion_body_model_units': position.tolist(),
        'controller_site_axes_in_quaternion_body_xyzw': rotation.as_quat().tolist(),
        'fixed_path_names': [x.get('name') for x in reversed(path)],
        'observation_frame': 'origin at position site; axes parallel to quaternion body',
        'controller_frame': 'origin and axes both at position site',
        'observation_to_controller_orientation': 'R_world_site = R_world_observation * R_body_site',
        'left_relative_rotation_rule': 'R_future_site * R_current_site.T = R_future_body * R_current_body.T',
        'cross_robot_tool_alignment_proven': False}


def controller_quaternion(observation_xyzw, definition):
    """Convert virtual observation axes to controller axes; origin unchanged."""
    if definition.get('format') != 'libero-eef-observation-frame-definition-v1':
        raise ValueError('Explicit EEF observation frame definition required')
    q = np.asarray(observation_xyzw, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q) - 1) > 1e-3:
        raise ValueError('Unit observation xyzw required')
    return (Rotation.from_quat(q) * Rotation.from_quat(
        definition['controller_site_axes_in_quaternion_body_xyzw'])).as_quat()
