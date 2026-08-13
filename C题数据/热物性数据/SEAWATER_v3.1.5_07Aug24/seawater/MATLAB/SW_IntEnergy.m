function u = SW_IntEnergy(T,uT,S,uS,P,uP)
    % SW_IntEnergy    Specific internal energy of seawater
    %=========================================================================
    % USAGE:  u = SW_IntEnergy(T,uT,S,uS,P,uP)
    %
    % DESCRIPTION:
    %   Specific internal energy of seawater at atmospheric pressure (0.1 MPa) using the specific
    %   enthalpy, vapor pressure, and density correlations given by [1].
    %   Values at non-atmospheric pressures (P_sat < P < 12 MPa) is obtained using
    %   specific enthalpy, vapor pressure, and density correlations given in [2].
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
    %
    % OUTPUT:
    %   u = specific internal energy [J/kg]
    %
    %   Note: u will have the same dimensions as T, S and P
    %
    % VALIDITY: 10 < T < 120 C, 0 < S < 120 g/kg, P_sat < P < 12 MPa
    %
    % ACCURACY: 1.5% (estimated from deviation of enthalpy function)
    %
    % REVISION HISTORY:
    %   2009-12-18: Mostafa H. Sharqawy (mhamed@mit.edu), MIT
    %               - Initial version
    %   2012-06-06: Karan H. Mistry (mistry@alum.mit.edu), MIT
    %               - Allow T,S input in various units
    %               - Allow T,S to be matrices of any size
    %   2015-04-15: Kishor G. Nayar (kgnayar@mit.edu), MIT
    %               - Extended function to 12 MPa
    %   2016-04-10: Karan H. Mistry (mistry@alum.mit.edu), MIT
    %               - Allow T,S to be matrices of any size
    %
    %   2024-08-07: John H. Lienhard (lienhard@mit.edu), MIT
    %               - Correct units passed to SW_Density and SW_Enthalpy
    %               - (Thanks to Tim Hatrell)
    %
    % DISCLAIMER:
    %   This software is provided "as is" without warranty of any kind.
    %   See the file sw_copy.m for conditions of use and licence.
    %
    % REFERENCES:
    %   [1] M. H. Sharqawy, J. H. Lienhard V, and S. M. Zubair, Desalination
    %       and Water Treatment, 16, 354-380, 2010. (http://web.mit.edu/seawater/)
    %   [2] K.G. Nayar, M. H. Sharqawy, L.D. Banchik and J. H. Lienhard V, Desalination,
    %       390, 1-24, 2016. (http://web.mit.edu/seawater/) 
    %=========================================================================

    %% CHECK INPUT ARGUMENTS

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


    % CHECK THAT S & T ARE WITHIN THE FUNCTION RANGE
    if ~isequal((T<10)+(T>120),zeros(size(T)))
        warning('Temperature is out of range for internal energy function 10<T<120 C');
    end

    if ~isequal((S<0)+(S>120),zeros(size(S)))
        warning('Salinity is out of range for internal energy function 0<S<120 g/kg');
    end

    Psat = SW_Psat(T,'C',S,'ppt')/1E6;

    if ~isequal((P<Psat)+(Psat>12),zeros(size(P)))
        warning('Pressure is out of range for internal energy function P_sat < P < 12 MPa');
    end

    %% BEGIN

    % rho = SW_Density(T,'C',S,uS,P,uP);
    % u  = SW_Enthalpy(T,'C',S,uS,P,uP) - ((P*1E6)./rho);
    % unit passing corrected, 2024/08/07
    rho = SW_Density(T,'C',S,'ppt',P,'MPa');
    u = SW_Enthalpy(T,'C',S,'ppt',P,'MPa') - ((P*1E6)./rho);

end
