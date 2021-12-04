

def make_empty_event():

    # Initialze variables to be returned
    hits = {'particle_id': [], 'layer': [], 'x': [], 'y': [], 'z': [], 't': [], 'hit_id': []}
    truth = {'hit_id': [], 'particle_id': [], 'weight': []}
    particles = {'vx': [], 'vy': [], 'vz': [], 'px': [], 'py': [], 'pz': [], 'q': [], 'nhits': [], 'particle_id': []}
    hit_info = []

    return hits, truth, particles, hit_info
