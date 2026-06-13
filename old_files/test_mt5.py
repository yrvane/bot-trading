import MetaTrader5 as mt5

# Initialiser la connexion
if not mt5.initialize():
    print(f"❌ Échec de connexion à MT5, erreur : {mt5.last_error()}")
else:
    print("✅ MT5 connecté avec succès !")
    
    # Afficher la version de MT5
    version = mt5.version()
    print(f"Version MT5 : {version}")
    
    # Récupérer les symboles disponibles qui correspondent à tes paires
    symbols = mt5.symbols_get()
    symbol_names = [s.name for s in symbols if "100" in s.name or "500" in s.name or "XAU" in s.name or "NAS" in s.name or "SPX" in s.name]
    
    print(f"\n🔍 Symboles trouvés correspondant à NASDAQ/SP500/GOLD :")
    for name in symbol_names[:20]:  # limite à 20 pour la lisibilité
        print(f"   - {name}")
    
    # Fermer la connexion
    mt5.shutdown()
    print("\n✅ Connexion MT5 fermée")