function mu_w = SW_ChemPot_w(T,uT,S,uS,P, uP)
    % SW_ChemPot_w    Chemical potential of water in seawater
    %=========================================================================
    % USAGE:  mu_w = SW_ChemPot_w(T,uT,S,uS,P,uP)
    %
    % DESCRIPTION:
    %   Chemical potential of water in seawater for pressures, P_sat < P < 12 MPa using
    %   thermodynamic relations and the specific Gibbs energy function given in [1].
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
    %   mu_w = chemical potential of water  [J/kg]
    %
    %   Note: mu_w will have the same dimensions as T and S
    %
    % VALIDITY: (1) 10 < T < 80 C; 0 < S < 120 g/kg; 0.1 < P = P0 < 1 MPa
    %           (2) 10 < T < 40 C; 0 < S < 42 g/kg; P_sat < P < 12 MPa
    %
    % ACCURACY: (1) 0.09 kJ/kg
    %           (2) 0.09 kJ/kg
    %
    % REVISION HISTORY:
    %   2009-12-18: Mostafa H. Sharqawy (mhamed@mit.edu), MIT
    %               - Initial version
    %   2012-06-06: Karan H. Mistry (mistry@alum.mit.edu), MIT
    %               - Allow T,S input in various units
    %               - Allow T,S to be matrices of any size
    %   2014-12-04: Kishor G. Nayar (kgnayar@mit.edu), MIT
    %               - Made new function differentiating the new Gibbs seawater function      %
    %   2015-04-28: Kishor G. Nayar (kgnayar@mit.edu), MIT
    %               - Added pressure dependence and updated the coefficients    %
    %   2016-04-10: Karan H. Mistry (mistry@alum.mit.edu), MIT
    %               - Allow T,S to be matrices of any size
    %   2016-04-15: Kishor G. Nayar (kgnayar@mit.edu), MIT
    %               - Extended validity range to 0 g/kg
    %   2016-07-08: Kishor G. Nayar (kgnayar@mit.edu), MIT
    %               - Corrected equation for accurate matrix calculation  
    % DISCLAIMER:
    %   This software is provided "as is" without warranty of any kind.
    %   See the file sw_copy.m for conditions of use and licence.
    %
    % REFERENCES:
    %   [1] K.G. Nayar, M. H. Sharqawy, L.D. Banchik and J. H. Lienhard V, Desalination,
    %       390, 1-24, 2016. (http://web.mit.edu/seawater/)    
    %=========================================================================

    %% CHECK INPUT ARGUMENTS

    % CHECK THAT S,T&P HAVE SAME SHAPE
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

    if ~isequal((T<10)+(T>80),zeros(size(T)))
        warning('Temperature is out of range for the chemical potential of water function 10 < T < 80 C');
    end

    if ~isequal((S<0)+(S>120),zeros(size(S)))
        warning('Salinity is out of range for the chemical potential of water function 0<S<120 g/kg');
    end

    Psat = SW_Psat(T,'C',S,'ppt')/1E6;

    P0 = Psat;
    P0(find(T<100)) = 0.101325;

    if ~isequal((P<Psat)+(Psat>12),zeros(size(P)))
        warning('Pressure is out of range for the chemical potential of water function P_sat < P < 12 MPa');
    end

    if ~isequal((P>=P0).*((S>42)+(T>40)),zeros(size(S)))
        warning('Salinity is out of range for the chemical potential of water function 10 < T < 40 C; 0.1 < S < 42 g/kg; P_sat < P < 12 MPa');
    end

    %% BEGIN

    a_s_KGN = [
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
    
    dg_sds= zeros(size(S));
    dg_Pds= zeros(size(S)); 
    dgds=zeros(size(S));  

    dg_sds(find(S>0))= ...
        a_s_KGN(1 )*ones(size(S(find(S>0))))            + ...
        a_s_KGN(2 )*T(find(S>0))                      + ...
        a_s_KGN(3 )*T(find(S>0)).^2                   + ...
        a_s_KGN(4 )*2*S(find(S>0))                    + ...
        a_s_KGN(5 )*2*S(find(S>0)).*T(find(S>0))      + ...
        a_s_KGN(6 )*2*S(find(S>0)).*(T(find(S>0)).^2) + ...
        a_s_KGN(7 )*3*S(find(S>0)).^2                          + ...
        a_s_KGN(8 )*3*(S(find(S>0)).^2).*T(find(S>0))          + ...
        a_s_KGN(9 )*3*S(find(S>0)).^2.*(T(find(S>0)).^2)        + ...
        a_s_KGN(10)*4*S(find(S>0)).^3                           + ...
        a_s_KGN(11)*4*(S(find(S>0)).^3).*T(find(S>0))           + ...
        a_s_KGN(12)*(ones(size(S(find(S>0)))) + log(S(find(S>0))))     + ...
        a_s_KGN(13)*((ones(size(S(find(S>0)))) + log(S(find(S>0)))).*T(find(S>0)));    % [J/g]



    dg_Pds(find(S>0)) = (P(find(S>0))-P0(find(S>0))).*(...
        a_sw_P_KGN(1)*zeros(size(T(find(S>0)))) + ...
        a_sw_P_KGN(2)*zeros(size(T(find(S>0)))) + ...
        a_sw_P_KGN(3)*zeros(size(T(find(S>0)))) + ...
        a_sw_P_KGN(4)*zeros(size(T(find(S>0)))) + ...
        a_sw_P_KGN(5)*ones(size(T(find(S>0))))  + ...
        a_sw_P_KGN(6)*T(find(S>0))              + ...
        a_sw_P_KGN(7)*T(find(S>0)).^2           + ...
        a_sw_P_KGN(8)*T(find(S>0)).^3             ...
    );                 % [J/g]



    dgds= dg_sds + dg_Pds;    %[J/g]

    g = SW_Gibbs(T,'C',S,'ppt',P,'MPa');    % [J/g]
    mu_w = g - S.*dgds;                     % [J/kg]

end
