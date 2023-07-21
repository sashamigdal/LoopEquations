test_array=rand(2000000)
Rational.(test_array)

(filter!(x->x.den<=1000000, (x->rationalize(x; tol=1e-10) ).(test_array))|>union!)[1:1000000]

function genbeta(len::Int, maxden)
    test_array=rand(Int(floor(1.2*len)))

    res=(filter!(x->x.den<=maxden, (x->rationalize(x; tol=1e-8) ).(test_array))|>union!)

    while (length(res) < len)
        test_array=rand(Int(floor(1.2*len)))
        res=vcat(res, (filter!(x->x.den<=maxden, (x->rationalize(x; tol=1e-8) ).(test_array))|>union!))|>union
    end
    res[1:len]
end

aaa=genbeta(1000000, 100000)
sort(aaa, by=(x->x.den))
1+1

ration1=rationalize(1.2; tol=1e-4)
ration1.den