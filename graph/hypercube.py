from graph import Graph

class Hypercube(Graph):
    """
    This class is used to define a hypercube structure or ordered lattice in n-dimension
    """

    def __init__(self, length, dimension, pbc=True, next_nearest=False):
        """
        Construct a new hypercube

        Args:
            length: The length of the hypercube
            dimension: The dimension of the system
            pbc: True for hypercube with periodic boundary conditions or False for hypercube with open boundary conditions
            next_nearest: To include the next nearest neighbors interaction for J1-J2 model

        TODO:non-squared hypercube
        """
        Graph.__init__(self)
        self.length = length
        self.dimension = dimension
        self.num_points = self.length ** self.dimension
        self.pbc = pbc
        self.next_nearest = next_nearest

        self.adj_list = self._create_adj_list()
        if self.next_nearest:
            self.adj_list_next = self._create_adj_list_next()
            
        self.num_bonds, self.bonds = self._find_bonds()
        if self.next_nearest:
            self.num_bonds_next, self.bonds_next = self._find_bonds_next()

    def _create_adj_list(self):
        """
            Create adjacency list for each point in the hypercube
        """
        adj_list = [[] for i in range(self.num_points)]
        for p in range(self.num_points):
            p_coordinate = self._point_to_coordinate(p)
            for d in range(self.dimension):
                neighbor1 = list(p_coordinate)
                neighbor2 = list(p_coordinate)
                if self.pbc or (not self.pbc and p_coordinate[d] + 1 < self.length):
                    neighbor1[d] = (p_coordinate[d] + 1) % self.length
                    adj_list[p].append(self._coordinate_to_point(neighbor1))
                if self.pbc or (not self.pbc and p_coordinate[d] - 1 >= 0):
                    neighbor2[d] = (p_coordinate[d] - 1 + self.length) % self.length
                    adj_list[p].append(self._coordinate_to_point(neighbor2))

        return adj_list

    def _create_adj_list_next(self):
        """
            Create adjacency list of the next nearest neighbour for 
            each point in the hypercube.
            
            TODO: does not work for general dimension
        """
        adj_list = [[] for i in range(self.num_points)]

        if self.dimension == 1:
            for p in range(self.num_points):
                ## Convert to coordinate
                p_coordinate = self._point_to_coordinate(p)
                ## Next nearest neighbour
                for d in range(self.dimension):
                    neighbor1 = list(p_coordinate)
                    neighbor2 = list(p_coordinate)
                    if self.pbc or (not self.pbc and p_coordinate[d] + 2 < self.length):
                        neighbor1[d] = (p_coordinate[d] + 2) % self.length
                        adj_list[p].append(self._coordinate_to_point(neighbor1))
                    if self.pbc or (not self.pbc and p_coordinate[d] - 2 >= 0):
                        neighbor2[d] = (p_coordinate[d] - 2 + self.length) % self.length
                        adj_list[p].append(self._coordinate_to_point(neighbor2))

        else:
            ## Generate all possible next nearest neighbours coordinate
            binary = [format(a, "#0%db" % (self.dimension+2))[2:] for a in range(2 ** self.dimension)]
            array_directions = [[1 if b == '1' else -1 for b in bin] for bin in binary]

            for p in range(self.num_points):
                ## Convert to coordinate
                p_coordinate = self._point_to_coordinate(p)
                ## Next nearest neighbour
                for dir in array_directions:
                    neighbor = list(p_coordinate)
                    for d, dim in enumerate(dir):
                        process = False
                        if dim == 1 and (self.pbc or (not self.pbc and p_coordinate[d] + 1 < self.length)):
                            process = True
                            neighbor[d] = (p_coordinate[d] + 1) % self.length
                        if dim == -1 and (self.pbc or (not self.pbc and p_coordinate[d] - 1 >= 0)):
                            process = True
                            neighbor[d] = (p_coordinate[d] - 1 + self.length) % self.length
                        
                        if not process:
                            break
                    
                    ## Only accepted if process is true
                    if process:
                        adj_list[p].append(self._coordinate_to_point(neighbor))


        return adj_list

    def _find_bonds(self):
        """
            Create bonds for each point. Similar to adjacency list
            but no repetition is calculated.
        """
        num_bonds = 0
        bonds = []
        for i in range(self.num_points):
            for j in self.adj_list[i]:
                if j > i:
                    num_bonds += 1
                    bonds.append((i, j))

        return num_bonds, bonds

    def _find_bonds_next(self):
        """
            Create bonds for each point next nearest neighbors. 
            Similar to adjacency list but no repetition is calculated.
        """
        num_bonds = 0
        bonds = []
        for i in range(self.num_points):
            for j in self.adj_list_next[i]:
                if j > i:
                    num_bonds += 1
                    bonds.append((i, j))

        return num_bonds, bonds

    def _point_to_coordinate(self, point):
        """
            Convert a given point to a coordinate based on row-major order
        """
        assert point < self.num_points
        coordinate = []
        for i in reversed(range(self.dimension)):
            v = self.length ** i
            coordinate.append(point // v)
            point = point % v

        return list(reversed(coordinate))

    def _coordinate_to_point(self, coordinate):
        """
            Convert a given coordinate to a point based on row-major order
        """
        assert len(coordinate) == self.dimension
        point = 0
        for i in range(self.dimension):
            point += coordinate[i] * (self.length ** i)

        return point

    def to_xml(self):
        stri = ""
        stri += "<graph>\n"
        stri += "\t<type>hypercube</type>\n"
        stri += "\t<params>\n"
        stri += "\t\t<length>%d</length>\n" % self.length
        stri += "\t\t<dimension>%d</dimension>\n" % self.dimension
        stri += "\t\t<pbc>%s</pbc>\n" % str(self.pbc)
        stri += "\t\t<next_nearest>%s</next_nearest>\n" % str(self.next_nearest)
        stri += "\t</params>\n"
        stri += "</graph>\n"
        return stri
