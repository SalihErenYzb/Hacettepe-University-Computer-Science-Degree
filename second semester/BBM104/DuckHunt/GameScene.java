import java.util.ArrayList;

import javafx.scene.Cursor;
import javafx.scene.Scene;
import javafx.scene.Group;

public class GameScene extends Scene {
	private ArrayList<DuckVerticalFlyPane> vert = new ArrayList<DuckVerticalFlyPane>();
	private ArrayList<DuckCrossFlyPane> cross = new ArrayList<DuckCrossFlyPane>();
	GameScene(){
    	super(new Group() ,DuckHunt.SCALE*256,DuckHunt.SCALE*240);
		vert.add(new DuckVerticalFlyPane("duck_black/",false,20*(Math.round(DuckHunt.SCALE))));
    	this.setRoot(new CustomLevelGroup(vert, cross, "level 1",GameStateManager.GameState.LEVEL2));

    	// Set cursor to crosshair icon when mouse enters game window
    	this.setOnMouseEntered(event -> {
    		DuckHunt.updateCursor();
    	    this.setCursor(DuckHunt.getCrosshairCursor());
    	});

    	// Set cursor to default icon when mouse leaves game window
    	this.setOnMouseExited(event -> {
    	    this.setCursor(Cursor.DEFAULT);
    	});
    	// Based on the gameState change manually create levels by adding birds with details to arraylist
    	// Then calling setRoot with level name in String and next level in GameState
	    DuckHunt.getGameStateManager().currentStateProperty()
	    .addListener((observable, oldState, newState) -> {
	    	if (newState == GameStateManager.GameState.LEVEL2) {
	    		cross.clear();
	    		vert.clear();
	    		cross.add(new DuckCrossFlyPane("duck_blue/", true,
	    				150*(Math.round(DuckHunt.SCALE)), false, 150*(Math.round(DuckHunt.SCALE))));
	    		cross.add(new DuckCrossFlyPane("duck_blue/", true,
	    				11*(Math.round(DuckHunt.SCALE)), false, 150*(Math.round(DuckHunt.SCALE))));	    		cross.add(new DuckCrossFlyPane("duck_blue/", true,
	    				150*(Math.round(DuckHunt.SCALE)), false, 80*(Math.round(DuckHunt.SCALE))));	    		cross.add(new DuckCrossFlyPane("duck_blue/", true,
	    				150*(Math.round(DuckHunt.SCALE)), false, 78*(Math.round(DuckHunt.SCALE))));	    		cross.add(new DuckCrossFlyPane("duck_blue/", true,
	    				75*(Math.round(DuckHunt.SCALE)), false, 94*(Math.round(DuckHunt.SCALE))));	    		cross.add(new DuckCrossFlyPane("duck_blue/", true,
	    				150*(Math.round(DuckHunt.SCALE)), false, 16*(Math.round(DuckHunt.SCALE))));	    		cross.add(new DuckCrossFlyPane("duck_blue/", true,
	    				47*(Math.round(DuckHunt.SCALE)), false, 150*(Math.round(DuckHunt.SCALE))));	    		cross.add(new DuckCrossFlyPane("duck_blue/", true,
	    				22*(Math.round(DuckHunt.SCALE)), false, 79*(Math.round(DuckHunt.SCALE))));	    		cross.add(new DuckCrossFlyPane("duck_blue/", true,
	    				22*(Math.round(DuckHunt.SCALE)), false, 130*(Math.round(DuckHunt.SCALE))));
	    		this.setRoot(new CustomLevelGroup(vert, cross, "level 2",GameStateManager.GameState.LEVEL3));
	    	}else if (newState == GameStateManager.GameState.LEVEL3) {
	    		
	    		cross.clear();
	    		vert.clear();
	    		vert.add(new DuckVerticalFlyPane("duck_black/",false,20*(Math.round(DuckHunt.SCALE))));
	    		vert.add(new DuckVerticalFlyPane("duck_blue/",true,45*(Math.round(DuckHunt.SCALE))));
	    		
	    		this.setRoot(new CustomLevelGroup(vert, cross, "level 3",GameStateManager.GameState.LEVEL4));
	    		
	    	}else if (newState == GameStateManager.GameState.LEVEL4) {
	    		
	    		

	    		cross.clear();
	    		vert.clear();
	    		cross.add(new DuckCrossFlyPane("duck_black/", false,
	    				30*(Math.round(DuckHunt.SCALE)), false, 5*(Math.round(DuckHunt.SCALE))));
	    		cross.add(new DuckCrossFlyPane("duck_blue/", true,
	    				160*(Math.round(DuckHunt.SCALE)), false, 50*(Math.round(DuckHunt.SCALE))));
	    		this.setRoot(new CustomLevelGroup(vert, cross, "level 4",GameStateManager.GameState.LEVEL5));
	    		
	    		
	    	}else if (newState == GameStateManager.GameState.LEVEL5) {
	    		cross.clear();
	    		vert.clear();
	    		cross.add(new DuckCrossFlyPane("duck_black/", false,
	    				30*(Math.round(DuckHunt.SCALE)), false, 5*(Math.round(DuckHunt.SCALE))));
	    		cross.add(new DuckCrossFlyPane("duck_red/", false,
	    				40*(Math.round(DuckHunt.SCALE)), true, 5*(Math.round(DuckHunt.SCALE))));
	    		cross.add(new DuckCrossFlyPane("duck_blue/", true,
	    				220*(Math.round(DuckHunt.SCALE)), false, 50*(Math.round(DuckHunt.SCALE))));
	    		

	    		
	    		this.setRoot(new CustomLevelGroup(vert, cross, "level 5",GameStateManager.GameState.LEVEL6));
	    		
	    		
	    	}else if (newState == GameStateManager.GameState.LEVEL6) {
	    		
	    		cross.clear();
	    		vert.clear();
	    		cross.add(new DuckCrossFlyPane("duck_black/", false,
	    				90*(Math.round(DuckHunt.SCALE)), true, 10*(Math.round(DuckHunt.SCALE))));
	    		cross.add(new DuckCrossFlyPane("duck_red/", false,
	    				60*(Math.round(DuckHunt.SCALE)), true, 20*(Math.round(DuckHunt.SCALE))));
	    		cross.add(new DuckCrossFlyPane("duck_blue/", false,
	    				60*(Math.round(DuckHunt.SCALE)), true, 50*(Math.round(DuckHunt.SCALE))));
	    		cross.add(new DuckCrossFlyPane("duck_blue/", true,
	    				80*(Math.round(DuckHunt.SCALE)), false, 40*(Math.round(DuckHunt.SCALE))));
	    		vert.add(new DuckVerticalFlyPane("duck_blue/",true,45*(Math.round(DuckHunt.SCALE))));


	    		
	    		this.setRoot(new CustomLevelGroup(vert, cross, "level 6",GameStateManager.GameState.LEVEL1));
	    		
	    		
	    	}
	    });
	}
}
