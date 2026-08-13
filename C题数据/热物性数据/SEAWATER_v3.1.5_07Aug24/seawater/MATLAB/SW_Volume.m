function v = SW_Volume(T,uT,S,uS,P,uP)
    % SW_Volume    Specific volume of seawater
    %=========================================================================
    % USAGE:  v = SW_Volume(T,uT,S,uS,P,uP)
    %
    % DESCRIPTION:
    %   Specific volume evaluated as inverse of density of seawater given in
    %   [1].
    %
    %
    % INPUT:
    %   T  = temperature
    %   uT = temperature unit
    %        'C'  : [degree Celsius] (ITS-90)
    %        'K'  : [Kelvin]
    %        'F'  : [degree Fahrenheit]
    %        'R'  : [Rankine]
    %   S  = salinity
    %   uS = salinity unit
    %        'ppt': [g/kg]  (reference-composition salinity)
    %        'ppm': [mg/kg] (in parts per million)
    %        'w'  : [kg/kg] (mass fraction)
    %        '%'  : [kg/kg] (in parts per hundred)
    %   P  = pressure
    %   uP = pressure unit
    %        'MPa': [MPa]
    %        'bar': [bar]
    %        'kPa': [kPa]
    %        'Pa' : [Pa]
    %
    %   Note: T, S and P must have the same dimensions
    %
    % OUTPUT:
    %   v = specific volume [m^3/kg]
    %
    %   Note: v will have the same dimensions as T, S and P
    %
    % VALIDITY: 0 < T < 180 C; 0 < S < 150 g/kg;
    %
    % ACCURACY: 0.1%
    %
    % REVISION HISTORY:
    %   2009-12-18: Mostafa H. Sharqawy (mhamed@mit.edu), MIT
    %               - Initial version based on density from [2]
    %   2012-06-06: Karan H. Mistry (mistry@alum.mit.edu), MIT
    %               - Allow T,S input in various units
    %               - Allow T,S to be matrices of any size
    %   2015-07-01: Kishor G. Nayar (kgnayar@mit.edu) and
    %               Adam Weiner (aweiner@mit.edu), MIT
    %               - Updated version based on [1]
    %   2016-04-10: Karan H. Mistry (mistry@alum.mit.edu), MIT
    %               - Allow T,S to be matrices of any size
    %
    % DISCLAIMER:
    %   This software is provided "as is" without warranty of any kind.
    %   See the file sw_copy.m for conditions of use and licence.
    %
    % REFERENCES:
    %   [1] K.G. Nayar, M. H. Sharqawy, L.D. Banchik and J. H. Lienhard V, Desalination,
    %       390, 1-24, 2016. (http://web.mit.edu/seawater/) 
    %   [2] M. H. Sharqawy, J. H. Lienhard V, and S. M. Zubair, Desalination
    %       and Water Treatment, 16, 354-380, 2010. (http://web.mit.edu/seawater/)
    %=========================================================================

    %% BEGIN

    rho = SW_Density(T,uT,S,uS,P,uP);
    v   = 1./rho;

end
