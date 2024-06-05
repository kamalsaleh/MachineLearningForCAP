







##
InstallMethod( CategoryOfParametrisedMorphisms,
          [ IsCapCategory ],
  
  function ( C )
    local name, Para;
    
    if not (IsStrictMonoidalCategory( C ) and IsSymmetricMonoidalCategory( C )) then
        Error( "the passed category must be a strict symmetric monoidal category" );
    fi;
    
    name := Concatenation( "CategoryOfParametrisedMorphisms( ", Name( C ), " )" );
    
    Para := CreateCapCategory( name,
             IsCategoryOfParametrisedMorphisms,
             IsObjectInCategoryOfParametrisedMorphisms,
             IsMorphismInCategoryOfParametrisedMorphisms,
             IsCapCategoryTwoCell
             : overhead := false );
    
    SetUnderlyingCategory( Para, C );
    
    ## Adding Operations
    ##
    AddObjectConstructor( Para,
      
      function ( Para, U )
        
        return CreateCapCategoryObjectWithAttributes( Para,
                  UnderlyingObject, U );
        
    end );
    
    ##
    AddObjectDatum( Para,
      
      function ( Para, A )
        
        return UnderlyingObject( A );
        
    end );
    
    ##
    AddIsWellDefinedForObjects( Para,
      
      function ( Para, A )
        
        return IsWellDefinedForObjects( UnderlyingCategory( Para ), UnderlyingObject( A ) );
        
    end );
    
    ##
    AddIsEqualForObjects( Para,
      
      function ( Para, A, B )
        
        return IsEqualForObjects( UnderlyingCategory( Para ),
                  UnderlyingObject( A ),
                  UnderlyingObject( B ) );
    end );
    
    # f_P:A -> B in Para is defined by an object P (in C) and a morphism f:P⊗A -> B (in C)
    AddMorphismConstructor( Para,
      
      function ( Para, source, datum, target )
        
        return CreateCapCategoryMorphismWithAttributes( Para,
                    source, target,
                    ParameterObject, datum[1],
                    ParametrisedMorphism, datum[2] );

    end );
    
    ##
    AddMorphismDatum( Para,
      
      function ( Para, f )
        
        return Pair( ParameterObject( f ), ParametrisedMorphism( f ) );
        
    end );
    
    ##
    AddIsWellDefinedForMorphisms( Para,
      
      function ( Para, f )
        local C, f_o, f_m, S, T;
        
        C := UnderlyingCategory( Para );
        
        f_o := ParameterObject( f );
        f_m := ParametrisedMorphism( f );
        
        S := UnderlyingObject( Source( f ) );
        T := UnderlyingObject( Target( f ) );
        
        return  IsWellDefinedForObjects( C, f_o ) and
                IsWellDefinedForMorphisms( C, f_m ) and
                IsEqualForObjects( C, Source( f_m ), TensorProductOnObjects( C, f_o, S ) ) and
                IsEqualForObjects( C, Target( f_m ), T );
    end );
    
    ##
    AddIsEqualForMorphisms( Para,
      
      function ( Para, f, g )
        
        C := UnderlyingCategory( Para );
        
        return  IsEqualForObjects( Para, Source( f ), Source( g ) ) and
                IsEqualForObjects( Para, Target( f ), Target( g ) ) and
                IsEqualForObjects( C, ParameterObject( f ), ParameterObject( g ) ) and
                IsEqualForMorphisms( C, ParametrisedMorphism( f ), ParametrisedMorphism( g ) );
        
    end );
    
    ##
    AddIsCongruentForMorphisms( Para,
      
      function ( Para, f, g )
        
        C := UnderlyingCategory( Para );
        
        return  IsEqualForObjects( Para, Source( f ), Source( g ) ) and
                IsEqualForObjects( Para, Target( f ), Target( g ) ) and
                IsEqualForObjects( C, ParameterObject( f ), ParameterObject( g ) ) and
                IsCongruentForMorphisms( C, ParametrisedMorphism( f ), ParametrisedMorphism( g ) );
        
    end );
    
    AddIdentityMorphism( Para,
      
      function ( Para, A )
        
        C := UnderlyingCategory( Para );
        
        return MorphismConstructor( Para,
                  A,
                  Pair( TensorUnit( C ), IdentityMorphism( C, UnderlyingObject( A ) ) ),
                  A );
        
    end );
    
    #
    #      P        Q
    #  S -----> U -----> T
    #      f        g
    #
    #          QxP
    #  S --------------> T
    #
    AddPreCompose( Para,
      
      function ( Para, f, g )
        local C, f_o, f_m, g_o, g_m;
        
        C := UnderlyingCategory( Para );
        
        f_o := ParameterObject( f );
        g_o := ParameterObject( g );
        
        f_m := ParametrisedMorphism( f );
        g_m := ParametrisedMorphism( g );
        
        return MorphismConstructor( Para,
                  Source( f ),
                  Pair( TensorProductOnObjects( C, g_o, f_o ), PreCompose( C, TensorProductOnMorphisms( C, IdentityMorphism( C, g_o ), f_m ), g_m ) ),
                  Target( g ) );
        
    end );
    
    AddAdditionForMorphisms( Para,
      
      function ( Para, f, g )
        local C;
        
        C := UnderlyingCategory( Para );
        
        if not IsEqualForObjects( C, ParameterObject( f ), ParameterObject( g ) ) then
              Error( "the two underlying objects must be equal!" );
        fi;
        
        return MorphismConstructor( Para,
                    Source( f ),
                    Pair( ParameterObject( f ), AdditionForMorphisms( C, ParametrisedMorphism( f ), ParametrisedMorphism( f ) ) ),
                    Target( f ) );
        
    end );
    
    AddSimplifyMorphism( Para,
      
      function ( Para, f, n )
        local C;
        
        C := UnderlyingCategory( Para );
        
        return MorphismConstructor( Para,
                  Source( f ),
                  Pair( ParameterObject( f ), SimplifyMorphism( C, ParametrisedMorphism( f ), n ) ),
                  Target( f ) );
        
    end );
    
    Finalize( Para );
    
    return Para;
    
end );

##
InstallOtherMethod( \/,
          [ IsCapCategoryMorphism, IsCategoryOfParametrisedMorphisms ],
  
  function ( f, Para )
    local C;
    
    C := CapCategory( f );
    
    Assert( 0, IsIdenticalObj( C, UnderlyingCategory( Para ) ) );
    
    return MorphismConstructor( Para,
                ObjectConstructor( Para, Source( f ) ),
                Pair( TensorUnit( C ), f ),
                ObjectConstructor( Para, Target( f ) ) );
    
end );

# Input:
#
#     P
#
# S -----> T
#     f
#
# a categorical implementation:
#
#                 S               0
#
# PreCompose( P -----> PxS, PxS -----> B )
#                 b               f
#
# where b = Braiding( S, P );
#
# a direct implementation:
#
InstallMethod( SwitchSourceAndParameterObject,
          [ IsMorphismInCategoryOfParametrisedMorphisms ],
  
  function ( f )
    local Para, C, P, S, h;
    
    Para := CapCategory( f );
    
    C := UnderlyingCategory( Para );
    
    P := ParameterObject( f );
    
    S := UnderlyingObject( Source( f ) );
    
    h := PreCompose( C, Braiding( C, S, P ), ParametrisedMorphism( f ) );
    
    return MorphismConstructor( Para,
              ObjectConstructor( Para, P ),
              Pair( S, h ),
              Target( f ) );
    
end );


# Input:
#
#     Q
#     |
#     |r
#     |
#     V
#     P
#
# S -----> T
#     f
#
# a categorical implementation:
#
#                         0                    P
#
# Switch( PreCompose( Q -----> P, Switch( S ------> T ) ) )
#                         r                    f
#
# a direct implementation:
#
InstallMethod( ReparametriseMorphism,
          [ IsMorphismInCategoryOfParametrisedMorphisms, IsCapCategoryMorphism ],
  
  function ( f, r )
    local Para, C, S, h;
    
    Para := CapCategory( f );
    
    C := UnderlyingCategory( Para );
    
    Assert( 0, IsIdenticalObj( C, CapCategory( r ) ) );
    
    S := UnderlyingObject( Source( f ) );
    
    h := PreCompose( C,
            TensorProductOnMorphisms( C, r, IdentityMorphism( C, S ) ),
            ParametrisedMorphism( f ) );
    
    return MorphismConstructor( Para,
              Source( f ),
              Pair( Source( r ), h ),
              Target( f ) );
    
end );

##
InstallMethod( NaturalEmbeddingIntoCategoryOfParametrisedMorphisms,
        [ IsCapCategory, IsCategoryOfParametrisedMorphisms ],
  
  function ( C, Para )
    local tau;
    
    Assert( 0, IsIdenticalObj( C, UnderlyingCategory( Para ) ) );
    
    tau := CapFunctor( "Natural embedding into category of parametrised morphisms", C, Para );
    
    AddObjectFunction( tau,
      
      function ( A )
        
        return ObjectConstructor( Para, A );
        
    end );
    
    AddMorphismFunction( tau,
      
      function ( source, f, target )
        
        return MorphismConstructor( Para, source, Pair( TensorUnit( C ), f ), target );
        
    end );
    
    return tau;
    
end );

##
InstallMethod( EmbeddingIntoCategoryOfParametrisedMorphisms,
        [ IsCategoryOfParametrisedMorphisms, IsCategoryOfParametrisedMorphisms ],
  
  function ( Para, Para_Lenses )
    local C, Lenses, iota, delta;
    
    C := UnderlyingCategory( Para );
    
    Lenses := UnderlyingCategory( Para_Lenses );
    
    iota := EmbeddingIntoCategoryOfLenses( C, Lenses );
    
    delta := CapFunctor( "Embedding into category of parametrised morphisms", Para, Para_Lenses );
    
    AddObjectFunction( delta,
      
      function ( A )
        
        return ObjectConstructor( Para_Lenses, ObjectConstructor(  Lenses, ListWithIdenticalEntries( 2, UnderlyingObject( A ) ) ) );
        
    end );
    
    AddMorphismFunction( delta,
      function ( source, f, target )
        
        return MorphismConstructor( Para_Lenses,
                  source,
                  Pair( ObjectConstructor(  Lenses, ListWithIdenticalEntries( 2, ParameterObject( f ) ) ), ApplyFunctor( iota, ParametrisedMorphism( f ) ) ),
                  target );
        
    end );
    
    return delta;
    
end );

##
InstallOtherMethod( \.,
          [ IsCategoryOfParametrisedMorphisms, IsPosInt ],
  
  function ( Para, string_as_int )
    local C, f, h;
    
    C := UnderlyingCategory( Para );
    
    if not IsCategoryOfSmoothMaps( C ) then
        TryNextMethod( );
    fi;
    
    f := NameRNam( string_as_int );
    
    if Int( f ) <> fail then
      
      return C.( f ) / Para;
      
    elif f = "LinearLayer" then
      
      return
        function ( m, n )
          local h, S, T, P;
          
          h := C.LinearLayer( m, n );
          
          S := ObjectConstructor( Para, ObjectConstructor( C, m ) );
          T := ObjectConstructor( Para, Target( h ) );
          
          P := ObjectConstructor( C, ( m + 1 ) * n );
          
          return MorphismConstructor( Para, S, Pair( P, h ), T );
          
        end;
        
    elif f in [ "IdFunc", "Sum", "Mul", "Power", "PowerBase", "Relu", "Sigmoid_", "Sigmoid", "Softmax_", "Softmax", "QuadraticLoss_",
                "QuadraticLoss", "CrossEntropyLoss_", "CrossEntropyLoss", "SoftmaxCrossEntropyLoss_", "SoftmaxCrossEntropyLoss" ] then
      
      return
        function ( arg... )
          
          return CallFuncList( C.( f ), arg ) / Para;
          
        end;
        
    elif f in [ "Sqrt", "Exp", "Log", "Sin", "Cos" ] then
        
        return C.( f ) / Para;
        
    else
        
        Error( "unrecognized-string!\n" );
        
    fi;
    
end );

##
InstallMethod( ViewString,
          [ IsObjectInCategoryOfParametrisedMorphisms ],
  
  function ( A )
    
    return ViewString( UnderlyingObject( A ) );
    
end );

##
InstallMethod( Display,
          [ IsObjectInCategoryOfParametrisedMorphisms ],
  
  function ( A )
    
    Print( "An object in ", Name( CapCategory( A ) ), " defined by:\n\n" );
    Display( UnderlyingObject( A ) );
    
end );

##
InstallMethod( ViewString,
          [ IsMorphismInCategoryOfParametrisedMorphisms ],
  
  function ( f )
    
    return
      Concatenation(
        ViewString( Source( f ) ),
        " -> ",
        ViewString( Target( f ) ),
        " defined by:",
        "\n\nParameter Object:\n-----------------\n",
        ViewString( ParameterObject( f ) ),
        "\n\nParametrised Morphism:\n----------------------\n",
        ViewString( ParametrisedMorphism( f ) ) );
    
end );

##
InstallMethod( DisplayString,
          [ IsMorphismInCategoryOfParametrisedMorphisms ],
  
  function ( f )
    
    return
      Concatenation(
        ViewString( Source( f ) ),
        " -> ",
        ViewString( Target( f ) ),
        " defined by:",
        "\n\nParameter Object:\n-----------------\n",
        ViewString( ParameterObject( f ) ),
        "\n\nParametrised Morphism:\n----------------------\n",
        DisplayString( ParametrisedMorphism( f ) ) );
    
end );

##
InstallMethod( Display,
          [ IsMorphismInCategoryOfParametrisedMorphisms ],
  
  function ( f )
    
    Print(
      Concatenation(
        ViewString( Source( f ) ),
        " -> ",
        ViewString( Target( f ) ),
        " defined by:",
        "\n\nParameter Object:\n-----------------\n",
        ViewString( ParameterObject( f ) ),
        "\n\nParametrised Morphism:\n----------------------\n" ) );
    
    Display( ParametrisedMorphism( f ) );
    
end );
