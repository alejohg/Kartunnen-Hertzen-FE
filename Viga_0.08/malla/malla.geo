h = 0.4;

// Puntos esquinas
Point(1)  = {0, 0, 0, 1};
Point(2)  = {5, 0, 0, 1};
Point(3)  = {5, h, 0, 1};
Point(4)  = {0, h, 0, 1};

// Punto carga puntual
Point(5) = {4, h, 0, 0.2};


// Puntos de control en bordes
Point(6) = {0, h/2, 0, 1};
Point(7) = {5, h/2, 0, 1};
Point(8) = {2.5, 0, 0, 1};
Point(9) = {2.5, h, 0, 1};
Point(10) = {2.5, h/2, 0, 1};


Line(1) = {1, 8};
//+
Line(2) = {8, 2};
//+
Line(3) = {2, 7};

Line(4) = {7, 3};

Line(5) = {3, 5};

Line(6) = {5, 9};

Line(7) = {9, 4};

Line(8) = {4, 6};

Line(9) = {6, 1};

Line(10) = {6, 10};

Line(11) = {10, 7};

Line(12) = {8, 10};

Line(13) = {10, 9};


Curve Loop(1) = {1, 2, 3, 4, 5, 6, 7, 8, 9};
//+
Plane Surface(1) = {1};
//+
Line{10, 11, 12, 13} In Surface{1}; 
Point{10} In Surface{1};

 
// Entidades fisicas 
//+
Physical Point("puntuales") = {5};
//+
Physical Curve("distr") = {5, 6, 7};

Physical Curve("empotramiento") = {3, 4, 8, 9};


Physical Surface("superf") = {1};


