function mu_s = SW_ChemPot_s(T,uT,S,uS,P,uP)

    % SW_ChemPot_s    Chemical potential of salt in seawater
    %=========================================================================
    % USAGE:  mu_s = SW_ChemPot_s(T,uT,S,uS,P,uP)
    %
    % DESCRIPTION:
    %   Chemical potential of salt in seawater for pressures, P_sat < P < 12 MPa using
    %   thermodynamic relations and the specific Gibbs energy function given in [1].
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
    %   mu_s = chemical potential of salts [J/kg]
    %
    %   Note: mu_s will have the same dimensions as T and S
    %
    % VALIDITY: (1) 10 < T < 80 C; 0.1 < S < 120 g/kg; 0.1 < P = P0 < 1 MPa
    %           (2) 10 < T < 40 C; 0.1 < S < 42 g/kg; P_sat < P < 12 MPa
    %
    % ACCURACY: (1) 12.70 kJ/kg
    %           (2) 12.70 kJ/kg
    %
    % REVISION HISTORY:
    %   2009-12-18: Mostafa H. Sharqawy (mhamed@mit.edu), MIT
    %               - Initial version
    %   2012-06-06: Karan H. Mistry (mistry@alum.mit.edu), MIT
    %               - Allow T,S input in various units
    %               - Allow T,S to be matrices of any size
    %   2015-07-01: Kishor G. Nayar (kgnayar@mit.edu), MIT
    %               - Revised function derived from new Gibbs seawater function
    %               - Included effect of pressure
    %   2016-04-04: Karan H. Mistry (mistry@alum.mit.edu), MIT
    %               - Allow T,S to be matrices of any size
    %
    % DISCLAIMER:
    %   This software is provided "as is" without warranty of any kind.
    %   See the file sw_copy.m for conditions of use and licence.
    %
    % REFERENCES:
    %   [1] K.G. Nayar, M. H. Sharqawy, L.D. Banchik and J. H. Lienhard V, Desalination,
    %       390, 1-24, 2016. (http://web.mit.edu/seawater/) 
    %=========================================================================


    %% CHECK INPUT ARGUMENTS

    % CHECK THAT S&T&P HAVE SAME SHAPE
    if ~isequal(size(S),size(T),size(P))
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

    % CONVERT SALINITY INPUT TO PPT
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

    %% CHECK THAT S, T & P ARE WITHIN THE FUNCTION RANGE

    if ~isequal((T<10)+(T>80),zeros(size(T)))
        warning('Temperature is out of range for the chemical potential of salt function 10 < T < 80 C');
    end

    if ~isequal((S<0.1)+(S>120),zeros(size(S)))
        warning('Salinity is out of range for the chemical potential of salt function 0.1<S<120 g/kg');
    end

    Psat = SW_Psat(T,'C',S,'ppt')/1E6;

    if ~isequal((P<Psat)+(Psat>12),zeros(size(P)))
        warning('Pressure is out of range for the chemical potential of salt function P_sat < P < 12 MPa');
    end

    %% Begin

    P0 = Psat;
    P0(find(T<100)) = 0.101325;

    if ~isequal((P>=P0).*((S>42)+(T>40)),zeros(size(S)))
        warning('Salinity is out of range for the chemical potential of salt function 10 < T < 40 C; 0.1 < S < 42 g/kg; P_sat < P < 12 MPa');
    end

    %% BEGIN
    %
    a_s_KGN=[
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

    dg_sds = ...
        a_s_KGN(1 )*ones(size(S))                   + ...
        a_s_KGN(2 )*T                               + ...
        a_s_KGN(3 )*T.^2                            + ...
        a_s_KGN(4 )*2*S                             + ...
        a_s_KGN(5 )*2*S.*T                          + ...
        a_s_KGN(6 )*2*S.*(T.^2)                     + ...
        a_s_KGN(7 )*3*S.^2                          + ...
        a_s_KGN(8 )*3*(S.^2).*T                     + ...
        a_s_KGN(9 )*3*S.^2.*(T.^2)                  + ...
        a_s_KGN(10)*4*S.^3                          + ...
        a_s_KGN(11)*4*(S.^3).*T                     + ...
        a_s_KGN(12)*(ones(size(S)) + log(S))        + ...
        a_s_KGN(13)*(ones(size(S)) + log(S)).*T;    % [J/g]

    a_sw_P_KGN = [
         9.9620e+02
         3.4910e-02
         4.7231e-03
        -6.9037e-06
        -7.2431e-01
         1.5712e-03
        -1.8919e-05
         2.5939e-08
    ];

    dg_Pds = (P-P0).*(...
        a_sw_P_KGN(1)*zeros(size(T)) + ...
        a_sw_P_KGN(2)*zeros(size(T)) + ...
        a_sw_P_KGN(3)*zeros(size(T)) + ...
        a_sw_P_KGN(4)*zeros(size(T)) + ...
        a_sw_P_KGN(5)*ones(size(T))  + ...
        a_sw_P_KGN(6)*T              + ...
        a_sw_P_KGN(7)*T.^2           + ...
        a_sw_P_KGN(8)*T.^3             ...
    );                 % [J/g]

    dgds = dg_sds + dg_Pds;    %[J/g]

    g = SW_Gibbs(T,'C',S,'ppt',P,'MPa');     % [J/g]
    mu_s = g + (1000-S).*dgds;               % [J/kg]

end
