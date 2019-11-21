import numpy as np
import numba as nb

def pairwize(array):
    return np.transpose(np.stack([np.roll(array, -i) for i in range(2)]))

def fitness(solution, distance):
    cost = 0
    pairs = pairwize(solution)
    for edge in pairs:
        cost += distance[tuple(edge)]
    return cost 

def get_travel(solution, city_points):
    travel = []
    for v in solution:
        travel.append(city_points[v])
    travel.append(city_points[solution[0]])
    return np.array(travel)

def compute_distances(cities):
    distances = np.zeros((len(cities), len(cities))) 
    for from_city, p1 in enumerate(cities):
        for to_city, p2 in enumerate(cities):
            if from_city != to_city:
                distances[from_city][to_city] = distances[to_city][from_city] = np.linalg.norm(p1-p2)
    return distances

def init_tsp(NB_CITIES):
    cities = np.random.rand(NB_CITIES, 2) * 100.0

    distances = np.zeros((NB_CITIES, NB_CITIES))

    for from_city, p1 in enumerate(cities):
        for to_city, p2 in enumerate(cities):
            if from_city != to_city:
                distances[from_city][to_city] = distances[to_city][from_city] = np.linalg.norm(p1-p2)
                
    #np.fill_diagonal(distances, 0.000000000000001)
    return cities, distances

def init_tsp_circular(NB_CITIES):
    cities = np.random.rand(NB_CITIES, 2)
    cities[:,0] = np.cos(np.linspace(0, 2*np.pi, NB_CITIES))
    cities[:,1] = np.sin(np.linspace(0, 2*np.pi, NB_CITIES))
    distances = np.zeros((NB_CITIES, NB_CITIES))

    for from_city, p1 in enumerate(cities):
        for to_city, p2 in enumerate(cities):
            if from_city != to_city:
                distances[from_city][to_city] = distances[to_city][from_city] = np.linalg.norm(p1-p2)
                
    #np.fill_diagonal(distances, 0.000000000000001)
    return cities, distances

def get_random_solution(NB_CITIES):
    solution=(list(range(NB_CITIES)))
    return np.random.shuffle(solution)
    
def plot_solution(solution, cities):
    import matplotlib.pyplot as plt
    from copy import deepcopy
    x, y = zip(*get_travel(solution, deepcopy(cities)))
    plt.plot(x,y)
    x, y = zip(*cities)
    plt.scatter(x, y, marker='*')
    
def plot_cities(cities, **kwargs):
    import matplotlib.pyplot as plt
    x, y = zip(*cities)
    plt.scatter(x, y, **kwargs)    