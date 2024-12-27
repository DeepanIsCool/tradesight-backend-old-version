def db_models(db, required_model):
    model_name = f"{required_model}_model"
    
    # Check if model already exists
    if model_name in db.Model.registry._class_registry:
        return db.Model.registry._class_registry[model_name]
    
    # Create new model class
    class StockModel(db.Model):
        __tablename__ = model_name
        __table_args__ = {'extend_existing': True}
        
        id = db.Column(db.Integer, primary_key=True)
        symbol = db.Column(db.String(10), unique=True, nullable=False)
        name = db.Column(db.String(100), nullable=False)

        def to_dict(self):
            return {
                'id': self.id,
                'symbol': self.symbol,
                'name': self.name,
            }
            
    StockModel.__name__ = model_name.title()
    return StockModel