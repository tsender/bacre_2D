# This message contains the description for creating an AutoSceneGenLandscape

float32 nominal_size    # The side-length [m or cm] of the nominal landscape along the X nad Y dimensions (this is a square landscape)
int32 subdivisions      # The number of times the two base triangles in the nominal landscape should be subdivided. 
                        # The landscape will have 2^NumSubdivisions triangles along each edge.
                        # Each vertex in the mesh will be spaced NomSize/(2^NumSubdivisions) [m or cm] apart in a grid.
float32 border          # (Optional) Denotes the approximate length to extend the nominal landscape in [m or cm]. 
                        # Using this will border the nominal landscape by ceil(Border/VertexSpacing) vertices in the four XY Cartesional directions, 
                        # where VertexSpacing is discussed above.