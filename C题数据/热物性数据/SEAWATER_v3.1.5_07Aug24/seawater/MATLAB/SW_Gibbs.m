function g = SW_Gibbs(T,uT,S,uS,P,uP)
    % SW_Gibbs    Specific gibbs energy of seawater
    %=========================================================================
    % USAGE:  g = SW_Gibbs(T,uT,S,uS,P,uP)
    %
    % DESCRIPTION:
    %   Specific Gibbs energy of seawater is obtained using the correlation
    %   given by [1]. Part of the correlation was fit against data from [2]
    %   and [3].
    %
    % INPUT:
    %   T  = temperature
    %   uT = temperature unit
    %        'C'  : [°C] (ITS-90)
    %        'K'  : [ K]
    %        'F'  : [°F]
    %        'R'  : [ R]
    %   S  = salinity
    %   uS = salinity unit
    %        'ppt': [ g/kg] (reference-composition salinity)
    %        'ppm': [mg/kg]
    %        'w'  : [kg/kg] (mass fraction)
    %        '%'  : [kg/kg] in parts per hundred
    %
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
    %   g = specific gibbs energy [J/kg]
    %
    %   Note: g will have the same dimensions as T, S and P
    %
    % VALIDITY: (1) 10 < T < 120 C; S = 0 g/kg; 0 < P < 12 MPa
    %           (2) 10 < T < 120 C; 0 < S < 120 g/kg; P = 0.1 MPa
    %           (3) 10 < T < 40 C; 0 < S < 42 g/kg; 0 < P < 12 MPa
    %           (4) 40 < T < 120 C; 0 < S < 42 g/kg; 0 < P < 12 MPa
    %           (5) 10 < T < 120 C; 42 < S < 120 g/kg; 0 < P < 12 MPa
    %
    % ACCURACY: (1) 0.07 kJ/kg
    %           (2) 0.07 kJ/kg
    %           (3) 0.07 kJ/kg
    %           (4) 0.11 kJ/kg (Extrapolated)
    %           (5) 0.11 kJ/kg (Extrapolated)
    %
    % REVISION HISTORY:
    %   2009-12-18: Mostafa H. Sharqawy (mhamed@mit.edu), MIT
    %               - Initial version based on [4]
    %   2012-06-06: Karan H. Mistry (mistry@alum.mit.edu), MIT
    %               - Allow T,S input in various units
    %               - Allow T,S to be matrices of any size
    %   2015-04-15: Kishor G. Nayar (kgnayar@mit.edu), MIT
    %               - Revised version with improved accuracy and valid up to 12 MPa
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
    %   [2] IAPWS release on the thermodynamic properties of seawater, 2008
    %   [3] IAPWS release on the Thermodynamic properties of ordinary water substance, 1996.
    %   [4] M. H. Sharqawy, J. H. Lienhard V, and S. M. Zubair, Desalination
    %       and Water Treatment, 16, 354-380, 2010. (http://web.mit.edu/seawater/)
    %=========================================================================

    %% CHECK INPUT ARGUMENTS
    %
    % CHECK THAT S,T & P HAVE SAME SHAPE
    if ~isequal(size(S),size(T), size(P))
        error('check_stp: S, T & P must have same dimensions');
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

    % CHECK THAT S, T & P ARE WITHIN THE FUNCTION RANGE
    if ~isequal((T<10)+(T>120),zeros(size(T)))
        warning('Temperature is out of range for Gibbs function 10<T<120 C');
    end

    if ~isequal((S<0)+(S>120),zeros(size(S)))
        warning('Salinity is out of range for Gibbs function 0<S<120 g/kg');
    end

    Psat = SW_Psat(T,'C',S,'ppt')/1E6;

    if ~isequal((P<Psat)+(Psat>12),zeros(size(P)))
        warning('Pressure is out of range for Gibbs function P_sat < P < 12 MPa');
    end

    %% BEGIN
    P0 = Psat;
    P0(find(T<100)) = 0.101325;

    a_w_KGN = [
         1.0677e+02
        -1.4303e+00
        -7.6139e+00
         8.3627e-03
        -7.8754e-06
    ];

    g_w_KGN = ...
        a_w_KGN(1)*ones(size(T)) +...
        a_w_KGN(2)*T +...
        a_w_KGN(3)*T.^2 +...
        a_w_KGN(4)*T.^3 +...
        a_w_KGN(5)*T.^4;


    a_sw_KGN = [
        -2.4176e+02
        -6.2462e-01
         7.4761e-03
         0
         1.3836e-03
        -6.7157e-06
         5.1993e-04
         0
         9.9176e-09
         0
         0
         6.6448e+01
         2.0681e-01
    ];

    g_s_KGN = zeros(size(S));
    g_s_KGN(find(S>0)) = ...
        a_sw_KGN(1 )*S(find(S>0))                                   + ...
        a_sw_KGN(2 )*S(find(S>0))   .*T(find(S>0))                  + ...
        a_sw_KGN(3 )*S(find(S>0))   .*T(find(S>0)).^2               + ...
        a_sw_KGN(4 )*S(find(S>0)).^2                                + ...
        a_sw_KGN(5 )*S(find(S>0)).^2.*T(find(S>0))                  + ...
        a_sw_KGN(6 )*S(find(S>0)).^2.*T(find(S>0)).^2               + ...
        a_sw_KGN(7 )*S(find(S>0)).^3                                + ...
        a_sw_KGN(8 )*S(find(S>0)).^3.*T(find(S>0))                  + ...
        a_sw_KGN(9 )*S(find(S>0)).^3.*T(find(S>0)).^2               + ...
        a_sw_KGN(10)*S(find(S>0)).^4                                + ...
        a_sw_KGN(11)*S(find(S>0)).^4.*T(find(S>0))                  + ...
        a_sw_KGN(12)*S(find(S>0)).*log(S(find(S>0)))                + ...
        a_sw_KGN(13)*S(find(S>0)).*log(S(find(S>0))).*T(find(S>0));

    a_sw_P_KGN=[
         9.961978E+02
         3.4910e-02
         4.7231e-03
        -6.9037e-06
        -7.2431e-01
         1.5712e-03
        -1.8919e-05
         2.5939e-08
    ];

    g_sw_P_KGN = (P-P0).*(              ...
        a_sw_P_KGN(1)*ones(size(T))   + ...
        a_sw_P_KGN(2)*T               + ...
        a_sw_P_KGN(3)*T.^2            + ...
        a_sw_P_KGN(4)*T.^3            + ...
        a_sw_P_KGN(5)*S               + ...
        a_sw_P_KGN(6)*S.*T            + ...
        a_sw_P_KGN(7)*S.*(T.^2)       + ...
        a_sw_P_KGN(8)*S.*(T.^3)         ...
    );

    g = (g_w_KGN + g_s_KGN + g_sw_P_KGN);

end
