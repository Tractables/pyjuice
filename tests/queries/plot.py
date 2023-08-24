import pyjuice as juice
import matplotlib.pyplot as plt
from pyjuice.nodes import multiply, summate, inputs
import pyjuice.nodes.distributions as dists

import pyjuice.visualize as juice_vis


def simple_pc_gen():
    n0 = inputs(0, num_nodes=256, dist=dists.Categorical(num_cats=5))
    n1 = inputs(1, num_nodes=256, dist=dists.Categorical(num_cats=3))
    n2 = inputs(2, num_nodes=256, dist=dists.Categorical(num_cats=2))

    m0 = multiply(n0, n1, n2)
    ns0 = summate(m0, num_nodes = 18)

    n3 = inputs(3, num_nodes=512, dist=dists.Categorical(num_cats=4))
    n4 = inputs(4, num_nodes=512, dist=dists.Categorical(num_cats=4))

    m1 = multiply(n3, n4)
    ns1 = summate(m1, num_nodes = 18)

    m2 = multiply(ns0, ns1)

    ns = summate(m2, num_nodes = 1)
    ns.init_parameters()
    return ns

ns = simple_pc_gen()

# case 1
plt.figure()
juice_vis.plot_pc(ns, node_id=True, node_num_label=True)
plt.show()

# case 2
juice_vis.plot_tensor_node_connection(ns, node_id=3)

# case 3
juice_vis.plot_tensor_node_connection(ns, node_id=4)
plt.show()

# case 4
juice_vis.plot_tensor_node_connection(ns, node_id=0)
