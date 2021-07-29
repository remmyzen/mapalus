from graph import Graph

class Network(Graph):
    """
    This class is used to define a structure from a networkx object
    """

    def __init__(self, network, pbc):
        """
        Construct a new graph

        Args:
            network: networkx graph
            pbc: True for hypercube with periodic boundary conditions or False for hypercube with open boundary conditions

        """
        Graph.__init__(self)
        self.network = network
        self.edges = self.network.edges
        self.num_points = len(self.network.nodes)
        self.pbc = pbc

        ## handle 1 dimension only integer
        if isinstance(list(self.network.nodes)[0], int):
            self.dimension = 1
        else: 
            self.dimension = len(list(self.network.nodes)[0])
        self.length = int(self.num_points ** (1.0/self.dimension))
        
        if self.dimension == 1:
            self.edges_temp = []
            for edge in self.edges:
                self.edges_temp.append((self._point_to_coordinate(edge[0]), self._point_to_coordinate(edge[1])))

            self.edges = self.edges_temp
    
        self.adj_list = self._create_adj_list()            
        self.num_bonds, self.bonds = self._find_bonds()

    def _create_adj_list(self):
        """
            Create adjacency list for each point in the network
        """
        adj_list = [[] for i in range(self.num_points)]
        for p in self.edges:
            from_point = self._coordinate_to_point(p[0])
            to_point = self._coordinate_to_point(p[1])
            adj_list[from_point].append(to_point)
            adj_list[to_point].append(from_point)

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
        stri += "\t<type>network</type>\n"
        stri += "\t<params>\n"
        stri += "\t\t<length>%d</length>\n" % self.length
        stri += "\t\t<dimension>%d</dimension>\n" % self.dimension
        stri += "\t\t<pbc>%s</pbc>\n" % str(self.pbc)
        stri += "\t</params>\n"
        stri += "</graph>\n"
        return stri
