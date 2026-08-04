# This is a set of functions that are used in SMM estimation. 
#The main function is fcn, which takes a vector of parameters and returns the vector of moments. 
#The other functions are used to build the parameter vector, compute test statistics, and print results.



function buildparam(p::Vector{Float64}) 
    # This function takes the vector of parameters p and builds the parameter subvector pea that is 
    #used in the moment generator.
    pea = zeros(pickmoments.nop); 
    
    pp = collect(p)
    pea[1] = pp[1];  
    pea[2] = pp[2];
    pea[3] = pp[3];
    pea[4] = pp[4];
    pea[5] = 0.10;

    return pea::Vector{Float64};   
  
end

# ============== The fcn function has two methods. One returns moments and basically wraps a wrapper. 
# ===============The other returns a GMM objective function. 
function fcn(p::Vector{Float64})
    pea = buildparam(p);
    moms = momentgen(pea);
    moms = moms[pickmoments.pick]
    return moms::Vector{Float64}
end 

function fcn(p::Vector{Float64},fopt::Float64);
    # This function takes a vector of parameters p and returns the GMM objective function value.

    pea = buildparam(p)
    
    simmoms     = collect(momentgen(pea)); #This is the vector of simulated moments. `momentgen` is the function that 
                                           #solves the model, simulates it, and generates the moments.

    if simmoms[1] != -100.0  #This part reads in the data moments, the covariance matrix of the moments, 
                             #and the names of the moments and parameters.
                             # -100.0 is a flag that indicates that the model did not solve. 
                             #If the model did not solve, the function returns a large value for the GMM objective function. 
        datamoms    = collect(readdlm("../Problemset2_2026/Output/moments.txt")); 
        sw          = collect(readdlm("../Problemset2_2026/Output/cluster_covariance_matrix.txt"));
        momname     = collect(readdlm("../Problemset2_2026/Output/momnames.txt"));
        pname       = collect(readdlm("../Problemset2_2026/Output/pnames.txt"));

        ch = pickmoments.pick; # This is a vector of indices that picks out the moments that are used in the estimation.
                               # Pickmoments is a structure defined elsewhere. 
        sw = sw[ch,ch]
        
        if settings.optweight #This conditional picks the weight matrix. If the user wants to use the identity matrix, 
                              #it does that. Otherwise, it uses the inverse of the covariance matrix of the moments.
            w = I(size(sw,1))
        else
            w = inv(sw);
        end
        datamoms = datamoms[ch]; 
        simmoms  = simmoms[ch]
        momdiff  = datamoms .- simmoms;
        momname  = momname[ch]

        bigQ     = momdiff'*w*momdiff;
        bigQ    = bigQ[1];

        #This part writes the progress to a file if the GMM function value is less than the optimal value.
        #It writes the data and simulated moments, the parameter values, and the GMM function value to a text file.
        if bigQ < fopt;
            filname = "Output/progress.txt"
            io = open(filname,"w")
            @printf(io," data and simulated moments\n")
            for jj in 1:size(momdiff,1)
                @printf(io,"%s",momname[jj]);
                @printf(io,"%16.8f",datamoms[jj])  
                @printf(io,"%16.8f\n",simmoms[jj])  
            end
            @printf(io," \n")
            @printf(io," parameter value\n")
            for jj in eachindex(p);
                @printf(io,"%s",pname[jj]);
                @printf(io,"%16.8f\n",p[jj])
            end
            @printf(io," \n")
            @printf(io,"GMM function value =   %16.8f\n",bigQ)
            close(io)
            sleep(2)
        end;
    else; 
        bigQ = globals.maxfunc #A very very large number that is returned if the model did not solve. 
                               #This is used to tell the optimizer to avoid this parameter vector.
    end

    return bigQ::Float64
end

function smmstats(p::Vector{Float64})
    # This function
    pea = buildparam(p)

    simmoms     = collect(momentgen(pea));

    #This part reads in the data moments, the covariance matrix of the moments, identifiers (gvkeys), 
    #and the influence functions. It then picks out the moments that are used in the estimation.
    datamoms    = collect(readdlm("../Problemset2_2026/Output/moments.txt")); 
    sw          = collect(readdlm("../Problemset2_2026/Output/cluster_covariance_matrix.txt"));
    gvkey       = collect(readdlm("../Problemset2_2026/Output/gvkeys.txt"));
    infl_fcn    = collect(readdlm("../Problemset2_2026/Output/influence_functions.txt"));

    infl_fcn    = infl_fcn[pickmoments.pick,:]

    #println(size(sw))
    ch = pickmoments.pick;
    sw = sw[ch,ch]
    datamoms = datamoms[ch]
    simmoms  = simmoms[ch]

    infl_fcn = infl_fcn .+ datamoms .- simmoms

    samplesize  = size(gvkey,1);

    if settings.optweight
        w = I(size(sw,1))
    else
        w = inv(sw);
    end

    dx          = jacobian(central_fdm(Int(pickmoments.nmom), 1) ,fcn, p)
    gee         = dx[1]
    

    gwg   = gee'*w*gee;                                               
    wsw   = w*sw*w;                                                   
    gwswg = gee'*wsw*gee;                                               

    igwg = inv(gwg);                                                   

    if settings.optweight                                              
       vc    =  igwg*gwswg*igwg;                                      
    else                                                              
       vc    =  igwg;                                                 
    end                                                             
  
    
    nsim = float((dims.nYears-dims.burnin)*dims.nFirms)/float(samplesize)
    vc    = (1.0 + 1.0/nsim)*vc # for code where you don't divide by n^2 in making the weight matrix, you divide by the sample size here


    standarderror = sqrt.(diag(vc));


    gigwgg = gee*igwg*gee';  
    eye = float(collect(I(size(w,1))))
    eyegg = eye - gigwgg*w; 
    vpe = eyegg*sw*eyegg'; 
    vpe = (1.0 + 1.0/nsim)*vpe # for code where you don't divide by n^2 in making the weight matrix, you divide by the sample size here
    ivpe = pinv(vpe);  # Moore Penrose
    
    momdiff = datamoms-simmoms
   
    if settings.optweight 
        jtest = momdiff'*ivpe*momdiff;  
    else
        jtest = momdiff'*w*momdiff; # for code where you don't divide by n^2 in making the weight matrix, you muliply by the sample size here
    end

    genshap =  igwg*gee'*w; 

    for jj = 1:pickmoments.nmom
        for ii = 1:pickmoments.noestp
            genshap[ii,jj] = genshap[ii,jj]
        end
    end
    infl_parameters = genshap*infl_fcn
    
   outtuple = (datamoms=datamoms, simmoms=simmoms, standarderror=standarderror, vpe=vpe, gee=gee, jtest=jtest, genshap=genshap, infl=infl_parameters)
   return outtuple 

end

function print_smm_results(smm_results,p)

    datamoms        = smm_results.datamoms
    simmoms         = smm_results.simmoms
    standarderror   = smm_results.standarderror
    vpe             = smm_results.vpe
    gee             = smm_results.gee
    genshap         = smm_results.genshap
    jtest           = smm_results.jtest[1]
    infl            = smm_results.infl

    momname     = collect(readdlm("../Problemset2_2026/Output/momnames.txt"));
    pname       = collect(readdlm("../Problemset2_2026/Output/pnames.txt"));
    momname    = momname[pickmoments.pick]
    
    if settings.optweight;
        filname = "Output/identity_results"
    else
        filname = "Output/results.txt"
    end

    io = open(filname,"w")
    @printf(io,"Final SMM results in LaTeX format \n")
    @printf(io," \n")

    @printf(io,"Parameter estimates and standard errors \n")
    @printf(io," \n")    
    
    for jj in eachindex(p); @printf(io,"&  %s   ",pname[jj]); end;   @printf(io," \n") 
    for jj in eachindex(p); @printf(io,"&  %12.4f  ",p[jj]);  end;   @printf(io," \n")  
    for jj in eachindex(p); @printf(io,"&( %12.4f )",standarderror[jj]); end;   @printf(io," \n")  
    @printf(io," \n")  

    @printf(io,"Data and Simulated Moments and t-stats \n")
    @printf(io," \n")    
    for jj in eachindex(datamoms);
        tstat = (datamoms[jj]-simmoms[jj])/sqrt(vpe[jj,jj]);
        @printf(io,"%s & %12.4f & %12.4f & %12.4f \\\\ \n",momname[jj],datamoms[jj],simmoms[jj],tstat); 
    end;   
    @printf(io," \n")  
        
    @printf(io,"Overidentification test \n")
    @printf(io," \n")    

    @printf(io,"GMM J-test %12.4f",jtest);
    @printf(io," \n")  
    
    @printf(io," \n")  
    
    @printf(io,"Andrews Gentzgow Shapiro \n")
    @printf(io," \n")    


    @printf(io,"           ")
    for jj in 1:pickmoments.noestp; @printf(io,"&   %s   ",pname[jj]); end;   @printf(io," \n") 
    for ii in 1:pickmoments.nmom
    @printf(io,"%s",momname[ii])
    for jj in eachindex(p);   ;@printf(io,"&  %12.4f  ",genshap[jj,ii]);  end;   @printf(io," \n")  
    end;
 
    @printf(io," \n")
    @printf(io,"Jacobian matrix \n")
    @printf(io," \n")    
    for ii in 1:pickmoments.nmom; 
    for jj in 1:pickmoments.noestp; @printf(io,"&  %12.4f  ",gee[ii,jj]);  end;   @printf(io," \n")  
    end;
    close(io)

    filname = "Output/influence_functions.txt"
    io = open(filname,"w")
    for ii in eachindex(infl[:,1]);
        for jj in 1:pickmoments.noestp; @printf(io,"  %36.16f  ",infl[jj,ii]);  end;   @printf(io," \n")
    end;
    close(io)
end;