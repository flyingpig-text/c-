function pi = SW_OsmPress(T,uT,S,uS)
    % SW_OsmPress    Osmotic pressure of seawater
    %=========================================================================
    % USAGE:  pi = SW_OsmPress(T,uT,S,uS)
    %
    % DESCRIPTION:
    %   The osmotic pressure of seawater is calculated from osmotic coefficient
    %   using a relationship given in [1]. The expression and discussion for
    %   osmotic coefficient and pressure can be found in detail in [2].
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
    %
    %   Note: T and S must have the same dimensions
    %
    % OUTPUT:
    %   pi = osmotic pressure [MPa]
    %
    %   Note: pi will have the same dimensions as T and S
    %
    % VALIDITY: (1) 0 < T < 200 C; 10 < S < 120 g/kg
    %           (2) 0 < T < 120 C; 0 < S < 10 g/kg
    %
    % ACCURACY: (1) 2.57% (Correlation)  - same as osm. coeff.
    %           (2) 0.89% (Extrapolated) - same as osm. coeff.
    %
    % REVISION HISTORY:
    %   2009-12-18: Mostafa H. Sharqawy (mhamed@mit.edu), MIT
    %               - Initial version based on chemical potentials
    %   2012-06-06: Karan H. Mistry (mistry@alum.mit.edu), MIT
    %               - Allow T,S input in various units
    %               - Allow T,S to be matrices of any size
    %   2015-11-01: Kishor G. Nayar (kgnayar@mit.edu), MIT
    %              - Updated version based on osmotic coefficient
    %   2016-04-10: Karan H. Mistry (mistry@alum.mit.edu), MIT
    %               - Allow T,S to be matrices of any size
    %
    % DISCLAIMER:
    %   This software is provided "as is" without warranty of any kind.
    %   See the file sw_copy.m for conditions of use and licence.
    %
    % REFERENCES:
    %   [1] R. A. Robinson, R.H. Stokes, Electrolyte Solutions: Second
    %       Revised Edition, Dover Publications, Inc., 2012
    %   [2] K.G. Nayar, M. H. Sharqawy, L.D. Banchik and J. H. Lienhard V, Desalination,
    %       390, 1-24, 2016. (http://web.mit.edu/seawater/) 
    %=========================================================================

    %% CHECK INPUT ARGUMENTS

    % CHECK THAT S&T HAVE SAME SHAPE
    if ~isequal(size(S),size(T))
        error('check_stp: S & T must have same dimensions');
    end

    % CONVERT TEMPERATURE INPUT TO °C
    switch lower(uT)
        case 'c'
        case 'k'
            T = T - 273.15;
        case 'f'
            T = 5/9*(T-32);
        case 'r'
            T = 5/9*(T-491.67);
        otherwise
            error('Not a recognized temperature unit. Please use ''C'', ''K'', ''F'', or ''R''');
    end

    % CONVERT SALINITY TO PPT
    switch lower(uS)
        case 'ppt'
        case 'ppm'
            S = S/1000;
        case 'w'
            S = S*1000;
        case '%'
            S = S*10;
        otherwise
            error('Not a recognized salinity unit. Please use ''ppt'', ''ppm'', ''w'', or ''%''');
    end

    % RANGE CHECKING NOT REQUIRED SINCE OTHER PROPERTIES ARE CALLED

    %% BEGIN


    Phi = SW_OsmCoeff(T,'C',S,'ppt');
    T_K = T + 273.15;

    % Weighted mol. weight in g/mol from Millero et al.,  Deep Sea Res. I 55 (2008) 50-72.
    MW_sw = 31.4038218;
    R = 8.3144598;

    %Define molality as a function of salinity
    m_sum = 1000*S./((1000 - S)*MW_sw);

    P0 = SW_Psat(T,'C',S,'ppt')/1E6;
    P0(find(T<100)) = 0.101325;


    rho_w_kgm3 = SW_Density(T,'C',zeros(size(S)),'ppt',P0,'MPa');
    Pi_sw      = R*(Phi.*m_sum.*T_K.*rho_w_kgm3);
    pi         = Pi_sw/(10^6);

end
