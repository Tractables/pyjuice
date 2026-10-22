import pyjuice as juice


def test_len_counts_the_layers():
    pc = juice.compile(juice.structures.HMM(seq_length = 4, num_latents = 4, num_emits = 5), verbose = False)
    for group in [pc.input_layer_group, *pc.inner_layer_groups]:
        assert len(group) == group.num_layers == len(list(group)) >= 1
