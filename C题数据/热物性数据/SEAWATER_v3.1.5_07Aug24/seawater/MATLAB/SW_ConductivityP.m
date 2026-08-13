function k = SW_ConductivityP(T,uT,S,uS,P,uP)
    % SW_Conductivity    Pressure-dependent thermal conductivity of seawater
    %=========================================================================
    % USAGE:  k = SW_ConductivityP(T,uT,S,uS,P,uP)
    %
    % DESCRIPTION:
    %   Pressure dependent thermal conductivity of seawater is calculated
    %   using correlation given in [1]. The correlation was developed from
    %   a previous correlation for thermal conductivity of seawater valid
    %   at 0.1 MPa given in [2]. Pressure dependence was incorporated from
    %   pure water data [3] and pressure dependent conductivity for seawater
    %   was verified against data from a model given in [4].
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
    %        'ppt': [g/kg] (reference-composition salinity)
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
    %
    % OUTPUT:
    %   k = thermal conductivity [W/m-K]
    %
    %   Note: k will have the same dimensions as T, S and P
    %
    % VALIDITY: (1) 10 < T < 90 C; 0 < S < 120 g/kg;  P = P0
    %           (2) 10 < T < 60 C; 0 < S < 35 g/kg; 0.1 < P < 12 MPa
    %
    % ACCURACY: (1) 2.59%
    %           (2) 2.59%
    %
    %
    % REVISION HISTORY:
    %   2015-06-19: Mostafa H. Sharqawy (mhamed@mit.edu), MIT
    %               - Raw source code
    %   2016-03-02: Kishor G. Nayar (kgnayar@mit.edu) and
    %               - Completed implementation of source code
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
    %   [2] M.H. Sharqawy, Desalination, 313, 97–104, 2013.
    %   [3] IAPWS, Release on the IAPWS Formulation 2011 for the Thermal Conductivity of
    %       Ordinary Water Substance, 2011.
    %   [4] P. Wang and A. Anderko, Int. J. Thermophys., 33, 235-258, 2012.
    %=========================================================================

    %% CHECK INPUT ARGUMENTS

    % CHECK THAT S,T&P HAVE SAME SHAPE
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

    % CONVERT PRESSURE INPUT TO MPa
    switch lower(uP)
        case 'mpa'
        case 'bar'
            P = P/10;
        case 'kpa'
            P = P/1000;
        case 'pa'
            P = P/1E6;
        otherwise
            error('Not a recognized pressure unit. Please use ''MPa'', ''bar'', ''kPa'', or ''Pa''');
    end

    % CHECK THAT S & T ARE WITHIN THE FUNCTION RANGE
    if ~isequal((T<10)+(T>90),zeros(size(T)))
        warning('Temperature is out of range for pressure-dependent thermal conductivity function 10 < T < 90 C');
    end

    if ~isequal((S<0)+(S>120),zeros(size(S)))
        warning('Salinity is out of range for pressure-dependent thermal conductivity function 0 < S < 120 g/kg');
    end

    Psat = SW_Psat(T,'C',S,'ppt')/1E6;

    if ~isequal((P<Psat)+(Psat>12),zeros(size(P)))
        warning('Pressure is out of range for pressure-dependent thermal conductivity function P_sat < P < 12 MPa');
    end

    %% BEGIN

    T_star = (T + 273.15)/300;
    P_star = (P - 0.1)/(139.9);

    k_fw0 = 0.797015135*T_star.^(-0.193823894) - 0.251242021*T_star.^(-4.7166384) + 0.0964365893*T_star.^(-6.38463554) - 0.0326956491*T_star.^(-2.13362102);

    A = 13.464*T_star.^4 - 60.727*T_star.^3 + 102.81*T_star.^2 - 77.387*T_star + 21.942;
    k_fw = k_fw0.*(1 + A.* P_star);

    B = 0.00022;
    k_sw = k_fw./(B*S + 1);

    k = k_sw;

end
