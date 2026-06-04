def fixed_seed_positions(world_size, organism_count):
    cx = world_size // 2
    cy = world_size // 2
    return [[cx, cy] for _ in range(organism_count)]
