function af = SW_FlowExergy(T,uT,S,uS,P,uP,T0,S0,P0)
    % SW_FlowExergy    Specific flow exergy of seawater, J/kg
    %=========================================================================
    % USAGE:  af = SW_FlowExergy(T,uT,S,uS,P,uP,T0,S0,P0)
    %
    % DESCRIPTION:
    %   Specific flow exergy of seawater at atmospheric pressure (0.1 MPa) using
    %   specific enthalpy, specific entropy, and chemical potential correlations given in [1].
    %   Values at non-atmospheric pressures (P_sat < P < 12 MPa) is obtained using
    %   specific enthalpy, specific entropy, and chemical potential correlations given in [2].
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
    %   T0 = reference state temperature
    %   S0 = reference state salinity
    %   P0 = reference state pressure
    %
    %   Note: T, S and P must have the same dimensions      %
    %   Note: T0,S0 and P0 have the same units given for T, S and P (i.e. uT, uS and uP)
    %   Note: T0, S0 amd P0 equal to zero sets the reference state to 25C, 35 g/kg and
    %         0.101325 MPa.
    % OUTPUT:
    %   af = specific flow exergy [J/kg]
    %
    %   Note: af will have the same dimensions as T, S and P
    %
    % VALIDITY: (1) 10 < T < 80 C; 0 < S < 120 g/kg; P = 0.1 MPa
    %           (2) 10 < T < 40 C; 0 < S < 42 g/kg; P_sat < P < 12 MPa
    %
    % ACCURACY: (1) 0.11 kJ/kg  (estimated at average value within the range)
    %           (2) 0.11 kJ/kg
    %
    % VALIDITY OF REFERENCE STATES: Validity range of (T0,S0,P0)is same as that of (T,S,P),
    %           except that S0 is valid only up to 0.1 g/kg due to singularity as S0
    %           approaches 0 g/kg. Entry of 'zero value' for reference state is automatically
    %           read as a standard state. Thus, (T0,S0,P0)=(0,0,0) would lead to
    %           (T0,S0,P0)=(25,35,0.101325)    %
    %
    % REVISION HISTORY:
    %   2009-12-18: Mostafa H. Sharqawy (mhamed@mit.edu), MIT
    %               - Initial version
    %   2012-06-06: Karan H. Mistry (mistry@alum.mit.edu), MIT
    %               - Allow T,S input in various units
    %               - Allow T,S to be matrices of any size
    %   2015-07-01: Kishor G. Nayar (kgnayar@mit.edu) and Adam Weiner (aweiner@mit.edu), MIT
    %               - Converted from EES to VBA with range limited to chemical potential of water
    %   2016-04-10: Karan H. Mistry (mistry@alum.mit.edu), MIT
    %               - Allow T,S to be matrices of any size
    %   2016-04-15: Kishor G. Nayar (kgnayar@mit.edu) and Adam Weiner (aweiner@mit.edu), MIT
    %               - Added error messages to reflect validity range of function
    %   2016-07-08: Kishor G. Nayar (kgnayar@mit.edu), MIT
    %               - Extended validity to S = 0 g/kg
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

    % CHECK THAT S&T HAVE SAME SHAPE
    if ~isequal(size(S),size(T), size(P))
        error('check_stp: S, T & P must have same dimensions');
    end

    % CONVERT TEMPERATURE INPUT TO °C
    switch lower(uT)
        case 'c'
        case 'k'
            T = T - 273.15;
            T0 = T0 - 273.15;
        case 'f'
            T = 5/9*(T-32);
            T0 = 5/9*(T0-32);
        case 'r'
            T = 5/9*(T-491.67);
            T0 = 5/9*(T-491.67);
        otherwise
            error('Not a recognized temperature unit. Please use ''C'', ''K'', ''F'', or ''R''');
    end

    % CONVERT SALINITY TO PPT
    switch lower(uS)
        case 'ppt'
        case 'ppm'
            S = S/1000;
            S0 = S0/1000;
        case 'w'
            S = S*1000;
            S0 = S0*1000;
        case '%'
            S = S*10;
            S0 = S0*10;
        otherwise
            error('Not a recognized salinity unit. Please use ''ppt'', ''ppm'', ''w'', or ''%''');
    end


    % CONVERT PRESSURE INPUT TO MPa
    switch lower(uP)
        case 'mpa'
        case 'bar'
            P = P/10;
            P0=P0/10;
        case 'kpa'
            P = P/1000;
            P0=P0/1000;
        case 'pa'
            P = P/1E6;
            P0=P0/1E6;
        otherwise
            error('Not a recognized pressure unit. Please use ''MPa'', ''bar'', ''kPa'', or ''Pa''');
    end

    % CHECK THAT S, T & P ARE WITHIN THE FUNCTION RANGE

    if ~isequal((T<10)+(T>80),zeros(size(T)))
        warning('Temperature is out of range for flow exergy function 10 < T < 80 C');
    end

    if ~isequal((S<0)+(S>120),zeros(size(S)))
        warning('Salinity is out of range for flow exergy function 0 < S < 120 g/kg');
    end

    Psat = SW_Psat(T,'C',S,'ppt')/1E6;

    P_0 = Psat;
    P_0(find(T<100)) = 0.101325;

    if ~isequal((P<Psat)+(Psat>12),zeros(size(P)))
        warning('Pressure is out of range for flow exergy function P_sat < P < 12 MPa');
    end

    if ~isequal((P>=P_0).*((S>42)+(T>40)),zeros(size(S)))
        warning('Salinity is out of range for flow exergy function 10 < T < 40 C; 0 < S < 42 g/kg; P_sat < P < 12 MPa');
    end     



   % DEFAULT REFERENCE STATE

   if (T0==0)
       T0 = 25;
   end

   if (S0==0)
       S0 = 35;
   end

   if (P0==0)
       P0 = 0.101325;
   end

    if (S0<0.1)    
        warning('Reference salinity is out of allowed range for flow exergy function 0.1 < S0 < 120');
    end  

    % RANGE CHECKING NOT REQUIRED SINCE OTHER PROPERTIES ARE CALLED

    %% BEGIN
    h_sw      = SW_Enthalpy(T,'C',S,'ppt',P,'MPa');
    s_sw      = SW_Entropy(T,'C',S,'ppt',P,'MPa');

    % Restricted Dead State
    h_sw_star = SW_Enthalpy(T0*ones(size(T)),'C',S,'ppt',P0*ones(size(T)),'MPa');
    s_sw_star = SW_Entropy(T0*ones(size(T)),'C',S,'ppt',P0*ones(size(T)),'MPa');
    mu_w_star = SW_ChemPot_w(T0*ones(size(T)),'C',S,'ppt',P0*ones(size(T)),'MPa');
    Smu_s_star = SW_SChemPot_s(T0*ones(size(T)),'C',S,'ppt',P0*ones(size(T)),'MPa');

    % Total Dead State
    mu_w_0    = SW_ChemPot_w(T0*ones(size(T)),'C',S0*ones(size(S)),'ppt',P0*ones(size(T)),'MPa');
    S0mu_s_0    = SW_SChemPot_s(T0*ones(size(T)),'C',S0*ones(size(S)),'ppt',P0*ones(size(T)),'MPa');
    Smu_s_0    = (S0mu_s_0/S0).*S;

    af = (h_sw - h_sw_star)-(T0+273.15).*(s_sw-s_sw_star)...
        + (1-0.001*S).*(mu_w_star-mu_w_0)+0.001*(Smu_s_star-Smu_s_0);

end
